
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial' : (0, 0.01),    
    'posterior_rho_initial': (-10, 0.01),   
}



def transform_logstd(logstd):
    return 1e-6 + F.softplus(logstd, beta=1, threshold=20)


class Conv2d_BBB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, bias=True ):
    
        super(Conv2d_BBB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        
        #self.random_sample = False
        self.random_sample = True



        self.prior_mu    = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)



    def forward(self, x, sample=False):
    
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        #self.W_sigma = transform_logstd(self.W_rho)
        
        
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu  = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)

        if self.random_sample:                        
            act_var = 1e-16 + F.conv2d(x ** 2, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups)
            act_std = torch.sqrt(act_var)
            
            eps1 = torch.empty(act_mu.shape[0], 1, 1, 1).normal_(0, 1).to(x.device)
            eps2 = torch.empty(1,act_mu.shape[1],1,1).normal_(0, 1).to(x.device)
            eps3 = torch.empty(1,1,act_mu.shape[2],1).normal_(0, 1).to(x.device)
            eps4 = torch.empty(1,1,1,act_mu.shape[3]).normal_(0, 1).to(x.device)
            eps  = 0.25*(eps1+eps2+eps3+eps4)
            #breakpoint()
            return act_mu + act_std * eps
        else:
            return act_mu



class Conv2d_BBB2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, bias=True ):
    
        super(Conv2d_BBB2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        
        #self.random_sample = False
        self.random_sample = True



        self.prior_mu    = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial2']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)



    def forward(self, x, sample=False):
    
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        #self.W_sigma = transform_logstd(self.W_rho)
        
        
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu  = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)

        if self.random_sample:                        
            act_var = 1e-16 + F.conv2d(x ** 2, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups)
            act_std = torch.sqrt(act_var)
            
            eps1 = torch.empty(act_mu.shape[0], 1, 1, 1).normal_(0, 1).to(x.device)
            eps2 = torch.empty(1,act_mu.shape[1],1,1).normal_(0, 1).to(x.device)
            eps3 = torch.empty(1,1,act_mu.shape[2],1).normal_(0, 1).to(x.device)
            eps4 = torch.empty(1,1,1,act_mu.shape[3]).normal_(0, 1).to(x.device)
            eps  = 0.25*(eps1+eps2+eps3+eps4)
            #breakpoint()
            return act_mu + act_std * eps
        else:
            return act_mu




class Linear_BBB(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear_BBB, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()
        self.random_sample = True
        
        
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=False):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        #self.W_sigma = transform_logstd(self.W_rho)

        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            #self.bias_sigma = transform_logstd(self.bias_rho)            
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        
        if self.random_sample :        
            act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
            act_std = torch.sqrt(act_var)
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(x.device)
            return act_mu + act_std * eps

        else:
            return act_mu


 
def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log( (sig_p / (sig_q+1e-8)) + 1e-8) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl
       
    

class Conv2d_VI(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(Conv2d_VI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        
        #self.random_sample = False
        self.random_sample = True
        self.current_kl    = 0


        self.prior_mu              = priors['prior_mu']
        self.prior_sigma           = priors['prior_sigma']
        self.posterior_mu_initial  = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)




    def forward(self, x ):
        self.current_kl  = 0

        # kernel weight
        if self.random_sample:                        
            sigma_weight = torch.log1p(torch.exp(self.W_rho))
            eps_kernel   = torch.randn_like(sigma_weight)
            tmp_result   = sigma_weight * eps_kernel
            weight       = self.W_mu + tmp_result

            self.current_kl  = calculate_kl(self.prior_mu,self.prior_sigma,self.W_mu, sigma_weight) 

        else:
            weight       = self.W_mu 




        # bias weight
        if self.use_bias:            
            if self.random_sample:                        
                sigma_bias = torch.log1p(torch.exp(self.bias_rho))
                eps_bias   = torch.randn_like(sigma_bias)
                bias       = self.bias_mu + (sigma_bias * eps_bias)    
                self.current_kl += calculate_kl(self.prior_mu,self.prior_sigma,self.bias_mu, sigma_bias) 

            else:
                bias       = self.bias_mu  




        else:
            bias           = self.bias_mu  




        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)    
        return out










class Linear_VI(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear_VI, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()
        #self.random_sample = False
        self.random_sample = True
        
        
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)


    def forward(self, x):
        # kernel weight
        if self.random_sample:                        
            sigma_weight = torch.log1p(torch.exp(self.W_rho))
            eps_kernel   = torch.randn_like(sigma_weight)
            tmp_result   = sigma_weight * eps_kernel
            weight       = self.W_mu + tmp_result
        else:
            weight       = self.W_mu 


        # bias weight
        if self.use_bias:            
            if self.random_sample:                        
                sigma_bias = torch.log1p(torch.exp(self.bias_rho))
                eps_bias   = torch.randn_like(sigma_bias)
                bias       = self.bias_mu + (sigma_bias * eps_bias)    
            else:
                bias       = self.bias_mu  
        else:
            bias           = self.bias_mu  


        out = F.linear(x, weight, bias)
        return out


