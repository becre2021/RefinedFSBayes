
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


#init_mu = 0
priors = {
    # 'prior_mu': 0,
    # 'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.01),
    'posterior_rho_initial': (-6, 0.01), #v=3300 conv vi  ResNet 18 seems works 
    'posterior_rho_initial2': (-25, 0.01),   #v=3900 conv vi
}


def transform_logstd(logstd):
    return F.softplus(logstd, beta=1, threshold=20)

    


class Conv2d_VI(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, bias=True , scalelogstd=-6):
        super().__init__()
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



        self.posterior_mu_initial  = priors['posterior_mu_initial']
        self.posterior_rho_initial = (scalelogstd,0.01)

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

        # kernel weight
        if self.random_sample:                        
            sigma_weight =  transform_logstd(self.W_rho)
            eps_kernel   = torch.randn_like(sigma_weight)
            tmp_result   = sigma_weight * eps_kernel
            weight       = self.W_mu + tmp_result
        else:
            weight       = self.W_mu 


        # bias weight
        if self.use_bias:            
            if self.random_sample:                        
                sigma_bias = transform_logstd(self.bias_rho)
                eps_bias   = torch.randn_like(sigma_bias)
                bias       = self.bias_mu + (sigma_bias * eps_bias)    
            else:
                bias       = self.bias_mu  
        else:
            bias           = self.bias_mu  


        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)    
        return out






class Linear_VI(nn.Module):

    def __init__(self, in_features, out_features,   bias=True, scalelogstd = -6):
        super(Linear_VI, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.posterior_mu_initial  = priors['posterior_mu_initial']
        self.posterior_rho_initial = (scalelogstd,0.01) 

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


    def forward(self, x):
        # kernel weight
        if self.random_sample:                        
            sigma_weight = transform_logstd(self.W_rho)
            eps_kernel   = torch.randn_like(sigma_weight)
            tmp_result   = sigma_weight * eps_kernel
            weight       = self.W_mu + tmp_result
        else:
            weight       = self.W_mu 


        # bias weight
        if self.use_bias:            
            if self.random_sample:                        
                sigma_bias = transform_logstd(self.bias_rho)
                eps_bias   = torch.randn_like(sigma_bias)
                bias       = self.bias_mu + (sigma_bias * eps_bias)    
            else:
                bias       = self.bias_mu  
        else:
            bias           = self.bias_mu  


        out = F.linear(x, weight, bias)
        return out


