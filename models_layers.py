import random
import numpy as np
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
        
from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.kl import kl_divergence




def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD

def transform_logstd(logstd):
    return 1e-6 + F.softplus(logstd, beta=1, threshold=20)


def compute_kl(q_dist, p_dist, eps=1e-4):
    zitter_mat = torch.eye(q_dist.cov.shape[-1])[None,...].to(q_dist.mean.device)        
    q_mvn = MVN(loc = q_dist.mean, covariance_matrix = q_dist.cov + zitter_mat      )
    p_mvn = MVN(loc = p_dist.mean.detach().clone(), covariance_matrix = (p_dist.cov + zitter_mat).detach().clone())                
    #print('q_mvn_mean: {}, q_mvn_cov: {}'.format( q_dist.mean,q_dist.cov + zitter_mat   ))

    outs = AttrDict()
    outs.q_dist = q_mvn
    outs.p_dist = p_mvn        
    outs.kl = kl_divergence(q_mvn,p_mvn) 
    return outs    
    
    
class Sequential_bool(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs    



class BayesLinear_localreparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.

    reference: 
    https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py
      
    """
    #def __init__(self, n_in, n_out ):
    def __init__(self, n_in:int, n_out:int , bias=True, with_reg=False ):
        
        super(BayesLinear_localreparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.with_reg = with_reg
        
        # learnable variational parameters
        self.qW_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.01, 0.01))        
        self.qW_logstd = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-6, -5))
        self.qb_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-1e-8, 1e-8))
        self.qb_logstd = nn.Parameter(torch.Tensor(self.n_out).uniform_(-6, -5))
        
        # self.register_parameter('qW_mu', torch.Tensor(self.n_in, self.n_out).uniform_(-0.01, 0.01))        
        # self.register_parameter('qW_logstd', torch.Tensor(self.n_in, self.n_out).uniform_(-6, -5))        
        # self.register_parameter('qb_mu', torch.Tensor(self.n_in, self.n_out).uniform_(-1e-8, 1e-8))        
        # self.register_parameter('qb_logstd', torch.Tensor(self.n_in, self.n_out).uniform_(-6, -5))                
        
        # non-learnable prior parameters
        self.register_buffer('pW_mu',torch.Tensor(self.n_in, self.n_out).zero_())
        self.register_buffer('pW_logstd',torch.Tensor(self.n_in, self.n_out).uniform_(-6, -5))
        self.register_buffer('pb_mu',torch.Tensor(self.n_out).zero_())
        self.register_buffer('pb_logstd',torch.Tensor(self.n_out).uniform_(-6, -5))
                

        
    def forward(self, X, use_sample=False):
        if not use_sample:  # This is just a placeholder function            
            output = torch.mm(X, self.qW_mu) + self.qb_mu.expand(X.size()[0], self.n_out)
            kld = 0

        else:
            # std_w = 1e-6 + F.softplus(self.qW_logstd, beta=1, threshold=20)
            # std_b = 1e-6 + F.softplus(self.qb_logstd, beta=1, threshold=20)
            std_W = transform_logstd(self.qW_logstd)
            std_b = transform_logstd(self.qb_logstd)

            #act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
            act_W_mu = torch.mm(X, self.qW_mu)  # self.W_mu + std_w * eps_W
            act_W_std = torch.sqrt(torch.mm(X.pow(2), std_W.pow(2)).clamp_(min=1e-16))

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = Variable(self.qW_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
            eps_b = Variable(self.qb_mu.data.new(std_b.size()).normal_(mean=0, std=1))
            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.qb_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)

            kld = KLD_cost(mu_p=self.pW_mu, sig_p=transform_logstd(self.pW_logstd), mu_q=self.qW_mu, sig_q=std_W)  
            kld = kld + KLD_cost(mu_p=self.pb_mu, sig_p=transform_logstd(self.pb_logstd), mu_q=self.qb_mu,sig_q=std_b)

        if self.with_reg:
            return output,kld,0 
        else:
            return output        
            
            
    def extra_repr(self):        
        s = ('in_features={}, out_features={}, bias={}').format(self.n_in,self.n_out,self.bias)
        return s            


class BayesConv2d_localreparam(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, with_reg=False):
        super(BayesConv2d_localreparam, self).__init__()

        """2d Convolutaional Layer with localreparametrization trick
        reference: 
        https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/layers/BBB_LRT/BBBConv.py

        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)        
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.with_reg = with_reg

        
        #unit_mu = .01 #failed train for cnn2
        #unit_mu = .1
        unit_mu = 1.
        
        # learnable variational parameters
        self.qW_mu = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).uniform_(-unit_mu, unit_mu)) 
        self.qW_logstd = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).uniform_(-6, -5))
        self.qb_mu = nn.Parameter(torch.Tensor(self.out_channels).uniform_(-1e-8, 1e-8))
        self.qb_logstd = nn.Parameter(torch.Tensor(self.out_channels).uniform_(-6, -5))
        
        # non-learnable prior parameters
        self.register_buffer('pW_mu',torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).zero_())
        self.register_buffer('pW_logstd',torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).uniform_(-6, -5))
        self.register_buffer('pb_mu',torch.Tensor(self.out_channels).zero_())
        self.register_buffer('pb_logstd',torch.Tensor(self.out_channels).uniform_(-6, -5))

        
    #def forward(self, x, sample=True):
    def forward(self, x, use_sample=True):
        
        if not use_sample:
            act_mu = F.conv2d(x, self.qW_mu, self.qb_mu, self.stride, self.padding, self.dilation, self.groups)            
            output = act_mu
            kld = 0
            
        else:
            std_W = transform_logstd(self.qW_logstd)
            std_b = transform_logstd(self.qb_logstd)

            #bias_var = self.bias_sigma ** 2
            act_mu = F.conv2d(x, self.qW_mu, self.qb_mu, self.stride, self.padding, self.dilation, self.groups)
            act_var = 1e-16 + F.conv2d(x ** 2, std_W.pow(2), std_b.pow(2), self.stride, self.padding, self.dilation, self.groups).clamp_(min=1e-16)
            act_std = torch.sqrt(act_var)

            eps = torch.empty(act_mu.size()).normal_(0, 1).to(x.device)
            output = act_mu + act_std * eps 
            
            kld = KLD_cost(mu_p=self.pW_mu, sig_p=transform_logstd(self.pW_logstd), mu_q=self.qW_mu, sig_q=std_W)  
            kld = kld + KLD_cost(mu_p=self.pb_mu, sig_p=transform_logstd(self.pb_logstd), mu_q=self.qb_mu,sig_q=std_b)
                       
        if self.with_reg:
            return output,kld,0 
        else:
            return output                    

    
    def extra_repr(self):        
        s = ('{}, {}, kernel_size={}' ', stride={}').format(self.in_channels,
                                                            self.out_channels,
                                                            self.kernel_size,
                                                            self.stride)
        return s