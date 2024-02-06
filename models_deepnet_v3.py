import random
import numpy as np
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.utils import parameters_to_vector

from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers



## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## base utility 
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal as LowMVN
from torch.distributions.kl import kl_divergence


eps        = 1e-8
eps_zitter = 1e-8
eps_log    = 1e-8
eps_clamp  = 1e-8
eps_nan    = 1e-8 
eps_eigv   = 1e-8
eps_zeros  = 1e-8

nan_val = 0
posinf_val = 1e1
neginf_val = -1e1
nanstd_val = 1e-2

def _num_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def transform_logstd(logstd):
    return 1e-6 + F.softplus(logstd, beta=1, threshold=20)


def build_MVN_Low(params:list , asprior=True):
    qf_mean,qf_Lcov = params
    zitter_mat = eps_zitter*torch.ones_like(qf_mean).to(qf_mean.device)                    
    if asprior:
        qf_dist = LowMVN(qf_mean.detach().clone(), qf_Lcov.detach().clone(), zitter_mat.detach().clone() )                        
    else:
        qf_dist = LowMVN(qf_mean ,qf_Lcov, zitter_mat)                                
    return qf_dist


def build_MVN_Full(params:list , asprior=True):
    qf_mean,qf_cov = params
    zitter_mat = eps_zitter*torch.eye(qf_mean.shape[-1]).to(qf_mean.device)[None,...]                    
    if asprior:    
        qf_dist = MVN(qf_mean.detach().clone() , (qf_cov +  zitter_mat).detach().clone()   )                        
    else:
        qf_dist = MVN(qf_mean , qf_cov +  zitter_mat)                               
    return qf_dist



def build_MVN_Diag(params:list , asprior=True, eps_zitter=1e-4):
    qf_mean,qf_std = params
    zitter_mat = eps_zitter*torch.ones_like(qf_mean).to(qf_mean.device)                    
    if asprior:    
        qf_dist = Normal(qf_mean.detach().clone() , (qf_std +  zitter_mat).detach().clone()   )                        
    else:
        qf_dist = Normal(qf_mean , qf_std +  zitter_mat)                               
    return qf_dist

    
def build_MVN(params:list, asprior=True, Lowrank=True, Diag=False):
    if Diag:
        return  build_MVN_Diag(params,asprior)   
    else:
        return  build_MVN_Low(params,asprior) if Lowrank  else build_MVN_Full(params,asprior)        
    
def compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
    qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
    pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
    #breakpoint()
    return kl_divergence(qf_dist,pf_dist).mean(dim=0,keepdim=True)


def _num_parameters(model):
    return sum([p.numel() for p in model.parameters()])





## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## setup deep model
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

__all__ = [           
           'model_deep_list',
           '_setup_model_deep'
           ]


model_deep_list = ['resnet20','resnet18',                  
                   'b2resnet20','b2resnet18',
                   'fbresnet20','fbresnet18',
                   'bmafbresnet20','bmafbresnet20v10'
                   'bmafbresnet18','bmafbresnet18v10',                   
                  ]


def _setup_model_deep(configs: AttrDict) -> nn.Module:
    in_dim, out_dim = configs.model.in_dim, configs.model.out_dim
    hidden_dim = 128 if configs.model.hidden_dim is None else configs.model.hidden_dim


    ## -------------------------------------------
    ## map lines
    ## -------------------------------------------

    if configs.model.modelname == 'resnet20':
        model = ResNet_typeA(BasicBlock, [3,3,3], in_dim=in_dim, out_dim=out_dim)                
    if configs.model.modelname == 'resnet18':
        model = ResNet_typeB(BasicBlock, [2,2,2,2], in_dim=in_dim, out_dim=out_dim)        
 

    ## -------------------------------------------
    ## mfvi lines
    ## -------------------------------------------

    if configs.model.modelname == 'b2resnet20':
        model = BNN2_ResNet_typeA(BasicBlock, [3,3,3], in_dim=in_dim, out_dim=out_dim, scalelogstd=configs.model.scalelogstd)
    if configs.model.modelname == 'b2resnet18':
        model = BNN2_ResNet_typeB(BasicBlock, [2,2,2,2], in_dim=in_dim, out_dim=out_dim, scalelogstd=configs.model.scalelogstd)
 
        

    ## -------------------------------------------
    ## fsvi lines
    ## -------------------------------------------

    if configs.model.modelname == 'fbresnet20':   
        model = FBNN_ResNet_typeA(BasicBlock,
                                  [3,3,3],
                                  in_dim=in_dim, 
                                  out_dim=out_dim, 
                                  scalelogstd=configs.model.scalelogstd,
                                  priorstdout = configs.model.priorstdout,
        )
        
    if configs.model.modelname == 'fbresnet18':   
        model = FBNN_ResNet_typeB(BasicBlock,
                                  [2,2,2,2],
                                  in_dim=in_dim, 
                                  out_dim=out_dim, 
                                  scalelogstd=configs.model.scalelogstd,
                                  priorstdout = configs.model.priorstdout,
        )        

        
    ## -------------------------------------------
    ## proposed lines
    ## -------------------------------------------

    if configs.model.modelname == 'bmafbresnet20':    ## Last layer without jacobian computation          
        model = FBNN_ResNet_Pro_typeA(BasicBlock, 
                                    [3,3,3],
                                    in_dim=in_dim, 
                                    out_dim=out_dim, 
                                    scalelogstd=configs.model.scalelogstd,
                                    priorstdout = configs.model.priorstdout,
                                    temppeak = configs.model.temppeak if 'temppeak' in configs.model.keys() else 0.5,
        )

    #if configs.model.modelname == 'bmafbresnet18': 
    if configs.model.modelname == 'bmafbresnet18v10': 
        model = FBNN_ResNet_Pro_typeB(BasicBlock, 
                                    [2,2,2,2],
                                    in_dim=in_dim, 
                                    out_dim=out_dim, 
                                    scalelogstd=configs.model.scalelogstd,
                                    priorstdout = configs.model.priorstdout,
                                    temppeak = configs.model.temppeak if 'temppeak' in configs.model.keys() else 0.5,
        )




    return model 





## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## ***resnet block***
## resnet type A (the smaller number of parameters) : 20, 32, 44, 56, 110, 1202
## resnet type B (the larger number of parameters)  : 18, 34, 50,101, 152       
## ex) resnet 20: .27M parameters, resnet 18:  11.4M parameters

## ***reference*** 
## resnet type A: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py         
## resnet type B: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py           
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 


'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''




## --------------------------------------------------------------------------------------------- 
## init weights
## --------------------------------------------------------------------------------------------- 

def _weights_init(m):         
    #print('current_m: {}'.format(m))   
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        # if m.bias:    
        #     init.constant(m.bias, 0)
        try:
            if m.bias.any():
                init.constant(m.bias, 0)
        except:
            pass
            
    elif isinstance(m, nn.BatchNorm2d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=1e-3)
        try:
            if m.bias.any():
                init.constant(m.bias, 0)
        except:
            pass
    else:
        pass
    return



## --------------------------------------- ------------------------------------------------------ 
## Bathcnorm revision to be compatible with functorchs
## --------------------------------------------------------------------------------------------- 
class BatchNorm2d_v2(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,affine=True, track_running_stats=True):
        super(BatchNorm2d_v2, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d_v2(planes)  # self.bn1 = nn.BatchNorm2d(planes , track_running_stats=True)
        self.bn2 = BatchNorm2d_v2(planes)  # self.bn2 = nn.BatchNorm2d(planes , track_running_stats=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out
    
    

## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## MAP
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
class Model_Base(nn.Module):
    def __init__(self,in_dim=1,out_dim=10,hidden_dim = 100):    
        super(Model_Base,self).__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.deterministic_nn = None
        self.stochastic_nn = None

    def forward(self,x,use_sample=True):        
        raise NotImplementedError()
        
    def predict_train(self, x , xc, nfsample=1 ,p_params=None, configs=None):        
        if configs is not None:
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            reglambda = 1e-3
            nfsample = 1

        
        self.train()    
            
        # compute logits
        logits = self.forward(x) #(nb,nout)        
                
        # compute reg
        param_vec = parameters_to_vector(self.parameters())
        reg = 0.5 * torch.norm(param_vec, p = 2)
        outs = AttrDict()
        outs.logits = logits
        outs.reg = reg                
        return outs        
        
        
    def predict_eval(self, x , nfsample=1, topk=1):        
        self.eval()        
        logits = self.deterministic_nn(x)
        pred = F.softmax(logits,dim=-1)           
        
        outs = AttrDict()
        outs.pred = pred  #prob                       
        outs.logits = logits[None,...]   #(nfsample,nb,ydim)      
        return outs       
        
 
    def _num_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    
    
class ResNet_typeA(Model_Base):
    #def __init__(self, block, num_blocks, num_classes=10):
    def __init__(self, block, num_blocks, in_dim=3, out_dim=10):
        
        super(ResNet_typeA, self).__init__()
        self.in_planes = 16
        
        conv1 = nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, type='A')
        layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, type='A')
        layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, type='A')
        linear = nn.Linear(64, out_dim)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                            linear                            
                           ]       
        self.deterministic_nn = nn.Sequential(*deterministic_nn)        
        self.deterministic_nn.apply(_weights_init)        


    #def _make_layer(self, block, planes, num_blocks, stride):
    def _make_layer(self, block, planes, num_blocks, stride, type='A'):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)        
        

    def forward(self, x):
        out = self.deterministic_nn(x)
        return out


class ResNet_typeB(Model_Base):
    #def __init__(self, block, num_blocks, num_classes=10):
    def __init__(self, block, num_blocks, in_dim=3, out_dim=10):        
        super(ResNet_typeB, self).__init__()
        self.in_planes = 64
        
        conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  type='B')
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, type='B')
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, type='B')
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, type='B')

        linear = nn.Linear(512*block.expansion, out_dim)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            layer4,                            
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                            linear                            
                           ]       
        self.deterministic_nn = nn.Sequential(*deterministic_nn)        
        self.deterministic_nn.apply(_weights_init)        


    #def _make_layer(self, block, planes, num_blocks, stride):
    def _make_layer(self, block, planes, num_blocks, stride, type):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
           
            
    def forward(self, x):
        out = self.deterministic_nn(x)
        return out    
            
    


            
            
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## MFVI BNN LastLayers Resnet
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
class LastLayer_BNN_Base(nn.Module):
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(LastLayer_BNN_Base,self).__init__()
    
        self.deterministic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype 

    #def extra_repr(self):
    def __repr__(self):        
        msg =  'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]

        
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        #set the variance of variational distribution
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set the parameters of the prior distribution        
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

        return    

       

    

class BNN2_ResNet_typeA(LastLayer_BNN_Base):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8):
        super(BNN2_ResNet_typeA, self).__init__(scalelogstd)
        self.in_planes = 16
        
        conv1 = nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = BatchNorm2d_v2(16) 
        layer1 = self._make_layer(block, 16, num_blocks[0], stride=1 , type='A')
        layer2 = self._make_layer(block, 32, num_blocks[1], stride=2 , type='A')
        layer3 = self._make_layer(block, 64, num_blocks[2], stride=2 , type='A')                
        linear = nn.Linear(64, out_dim)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]       
        
        stochastic_nn    = [
                            linear
                           ]       
        

        self.deterministic_nn = nn.Sequential(*deterministic_nn)        
        self.stochastic_nn =  nn.Sequential(*stochastic_nn)       
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)
        self.setup_parameters()                                     # setup variance parameters
        

    def _make_layer(self, block, planes, num_blocks, stride, type='A'):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)        
        
        
    #def extra_repr(self):
    def __repr__(self):        
        msg = 'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd_s))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_s),_num_parameters(self.params_p_logstd_s))
        return msg[:-1]
        
    
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        self.params_q_logstd_s = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])         
        self.params_p_mean_s   = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd_s = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        return    
        
    
    def setup_functiondist(self):    
        func_model_s, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model_s =  lambda params, x_in : func_model_s(params, buffers, x_in) 
        self.flag_setup = True         
        return             
        
    def sample_weight(self, param_q_mean, param_q_logstd):        
        weight_sample = []
        for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
            ith_weight_sample = ith_q_mean + transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)  
            weight_sample.append(ith_weight_sample)
        return weight_sample
        
                        
    def compute_func_samples(self, x,  param_q_mean_s, param_q_logstd_s, nfsample=5):                
        self.setup_functiondist() if not self.flag_setup else 1
        h = self.deterministic_nn(x)                

        func_samples = []
        for _ in range(nfsample):
            ith_weight_sample_s = self.sample_weight(param_q_mean_s, param_q_logstd_s)                        
            ith_func_samples = self.func_model_s(ith_weight_sample_s, h)
            func_samples.append(  ith_func_samples[None,...]  )  
        func_samples = torch.cat(func_samples,dim=0) #(nfsample,nb,ydim)                   
        return func_samples
               
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = 1
            
        else:            
            nctset = 2 
            reglambda = 1e-8
            nfsample = 1
        
        
        self.train()                    
        # get variational params    
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x,  param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)

        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
                                
            # get prior params    
            param_p_mean_s, param_p_logstd_s    = list(self.params_p_mean_s) ,  list(self.params_p_logstd_s)       

            # compute kl on weight space
            params_qf = [param_q_mean_s,param_q_logstd_s]
            params_pf = [param_p_mean_s,param_p_logstd_s]             
            weight_kl = self.compute_weight_kl(params_qf, params_pf) / x.size(0)

        else :
            weight_kl = torch.tensor(0.0).to(x.device)            
            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = weight_kl               
   
        return outs        
       
    def compute_weight_kl(self, params_qf, params_pf, eps=1e-8):
        # get params
        param_q_mean_s,param_q_logstd_s = params_qf 
        param_p_mean_s,param_p_logstd_s = params_pf
        
        # compute kl on weight space        
        kl_list = []
        for i_q_m,i_q_logstd,i_p_m,i_p_logstd in zip(param_q_mean_s,param_q_logstd_s,param_p_mean_s,param_p_logstd_s):
            i_kl = compute_kl([i_q_m,  transform_logstd(i_q_logstd) ], 
                              [i_p_m,  transform_logstd(i_p_logstd) ] ,
                              Lowrank=False, Diag=True).sum().unsqueeze(0)
            kl_list.append(i_kl)
        #breakpoint()
        weight_kl = torch.cat(kl_list,dim=0).sum(dim=0) #(nparams)
                    
        return weight_kl
       
            
    def predict_eval(self, x , nfsample=1,topk=1):        
        self.eval()        
        # get variational params    
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x, param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample   #(nfsample,nb,ydim)      

        return outs       
    
    
    
    
class BNN2_ResNet_typeB(BNN2_ResNet_typeA):
#    def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8):

        super(BNN2_ResNet_typeB, self).__init__(block, num_blocks, in_dim, out_dim, scalelogstd)
        self.in_planes = 64
        
        conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  type='B')
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, type='B')
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, type='B')
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, type='B')

        linear = nn.Linear(512*block.expansion, out_dim)



        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            layer4,                            
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]       

        
        stochastic_nn    = [
                            linear
                           ]       
        

        self.deterministic_nn = nn.Sequential(*deterministic_nn)        
        self.stochastic_nn =  nn.Sequential(*stochastic_nn)       
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)
        self.setup_parameters()           
    
    






## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## Tractable Function Space BNN Resnet (T-FVI)
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

class FBNN_ResNet_typeA(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdin = 1.0, priorstdout = 10.):

        self.scalelogstd = scalelogstd        
        self.priorstdin  = priorstdin
        self.priorstdout = priorstdout        

        super(FBNN_ResNet_typeA, self).__init__()
        self.in_planes = 16        
        conv1 = nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)        
        bn1 = BatchNorm2d_v2(16) 
        layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)                
        linear = nn.Linear(64, out_dim)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,                    
                            layer2, 
                            layer3,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]       
        
        stochastic_nn    = [
                            linear
                           ]       
            

        self.deterministic_nn = nn.Sequential(*deterministic_nn)   
        self.stochastic_nn    =  nn.Sequential(*stochastic_nn)                       
        #self.mixture_module   = mixture_module     
        
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()
        
        
        
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean  = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        return    
    
    
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         


    #def extra_repr(self):
    def __repr__(self):        
        msg = 'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]

        

    def _make_layer(self, block, planes, num_blocks, stride, type='A'):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, type))
            self.in_planes = planes * block.expansion            
        return nn.Sequential(*layers)
        

    def sample_weight(self, param_q_mean, param_q_logstd, use_bayes=True):        
        weight_sample = []
        for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
            if use_bayes:
                ith_weight_sample = ith_q_mean + transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)                       
            else:
                ith_weight_sample = ith_q_mean
            weight_sample.append(ith_weight_sample)
        return weight_sample
    

    def compute_logit_samples(self, x ,use_bayes=True, configs=None):                
                
        self.train()            
        #nfsample = configs.train.nfsample if configs is not None else 1
        nfsample = 1
        
        # get hidden features
        h = self.deterministic_nn(x)        
        
        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     

        # compute function samples
        logits_sample=[]
        for _ in range(nfsample):
            ith_weight_sample = self.sample_weight(param_q_mean, param_q_logstd , use_bayes=use_bayes)            
            logits_sample.append(  self.func_model(ith_weight_sample , h)[None,...]  )  
        logits_sample = torch.cat(logits_sample,dim=0) #(nfsample,nb,ydim)                           
                    
        return logits_sample
    
    
    
    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)




    def compute_funcdist_jacobian(self, x , param_q_mean, param_q_logstd,  ):
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        h = self.deterministic_nn(x)
        f_mean = self.func_model(param_q_mean, h )  

        # compute f_Lcov
        with torch.no_grad():
            jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h )                
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec    = transform_logstd(param_q_logstd_vec) #(nparams)        
        f_Lcov = jacobian_f_vec * param_q_std_vec[None,None,:] #(nb,ydim,nparams)
        f_cov = f_Lcov @ f_Lcov.transpose(-1,-2) #(nb,ydim,ydim)
        return f_mean, f_cov, h


    ## compute the parameters of function space prior / posterior distirbution without the use of autograd
    def compute_funcdist_jacobian_wo(self, x , param_q_mean, param_q_logstd, no_grad_jacobian=True ):

        #self.setup_functiondist() if not self.flag_setup else 1        
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        h = self.deterministic_nn(x) 
        f_mean = self.func_model(param_q_mean, h ) 

        # use values of jacobian without grad
        if no_grad_jacobian:
            h = h.detach().clone()

        # comopute f_std
        param_q_std  = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov, h
    
    


    def compute_fdist(self, xc , p_params=None,configs=None):                
        self.train()                
    
        # compute "q" function space dist on context input        
        param_q_mean, param_q_logstd          = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        qf_mean_xc , qf_covdiag_xc, h_feature_xc  = self.compute_funcdist_jacobian_wo( xc, param_q_mean, param_q_logstd,no_grad_jacobian=True )            
        qf_std_xc                             = torch.sqrt(qf_covdiag_xc)

        
        # compute "p" function space dist on context input                
        pf_mean_xc =                     torch.zeros_like(qf_mean_xc).to(xc.device) #retry
        pf_std_xc  =    self.priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)     



        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        return param_qf, param_pf, None 
    
        
    
    
        
    def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
        def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
            kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
            kl_div = kl_div_qf_pf.sum(dim=-1)
            return kl_div #(nb,)
        
        
        # set params
        qf_mean,qf_cov = params_qf
        pf_mean,pf_cov = params_pf

        # split ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
            if weight is None:            
                i_weight_kl_batch = i_kl_batch                
            else:
                i_weight          = weight[i_set]                       
                i_weight_kl_batch = i_weight*i_kl_batch
            kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = (1 / num_batch)*kl_divergence_list.max()
        return torch.clamp(sup_kl,min=eps)    
    
        
    def predict_eval(self, x , nfsample=1, topk=1):        
        self.eval()
        # get hidden features
        h = self.deterministic_nn(x)        

        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     

        # compute function samples
        logits_sample=[]
        for _ in range(nfsample):
            ith_weight_sample = self.sample_weight(param_q_mean, param_q_logstd)            
            logits_sample.append(  self.func_model(ith_weight_sample , h)[None,...]  )  
        logits_sample = torch.cat(logits_sample,dim=0)  #(nfsample,nb,ydim)         
        pred          = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample   #(nfsample,nb,ydim)      
        return outs       

            
            
                       
           
class FBNN_ResNet_typeB(FBNN_ResNet_typeA):
    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdin = 1.0, priorstdout = 10.):

        self.scalelogstd = scalelogstd        
        self.priorstdin  = priorstdin
        self.priorstdout = priorstdout        

        super(FBNN_ResNet_typeB, self).__init__(block, num_blocks, in_dim, out_dim, scalelogstd, priorstdin, priorstdout)
        self.in_planes = 64        
        conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  type='B')
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, type='B')
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, type='B')
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, type='B')
        linear = nn.Linear(512*block.expansion, out_dim)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            layer4,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]               
        stochastic_nn    = [
                            linear
                           ]       


        self.deterministic_nn = nn.Sequential(*deterministic_nn)   
        self.stochastic_nn    =  nn.Sequential(*stochastic_nn)                       
        
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()
        
        
        
        


## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## Proposed Informative Function Space BNN Resnet (R-FVI)
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 


from torch.nn.utils import parameters_to_vector
from torch.autograd import Variable
        
from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal as LowMVN
from torch.distributions.kl import kl_divergence
from attrdict import AttrDict

from torch.distributions.categorical import Categorical



class Hiddenfeature_Mixture_Module(nn.Module):     
    def __init__(self, nfeature: int, nlabel: int , usediagcov=False, gamma=0.99,eps_zitter=1e-4):
        super(Hiddenfeature_Mixture_Module,self).__init__()
        
        self.nlabel = nlabel
        self.nfeature = nfeature
        self.eps_zitter = eps_zitter
        self.gamma = gamma
        self.usediagcov = usediagcov
        
        # per epoch        
        self.static_cnt_sum    =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.static_mean_sum   =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature), requires_grad=False )
        self.static_cov_sum    =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature, self.nfeature) , requires_grad=False )        
        return 

    
    #def __repr__(self):
    def extra_repr(self):        
        msg = 'nlabel: {}, \n'.format(self.nlabel)
        msg += 'nfeature: {}, \n'.format(self.nfeature)
        return msg[:-1]
    
           
       
    def update_layers_static_batch(self, h_feature, y, eps_zitter=1e-4):
        # set idx_per_label        
        idx_per_label =  [(y == k)  for k in range(self.nlabel) ]  
        
        # compute emirical mean and cov
        static_mean_sum, static_cov_sum = [],[]        
        for idx_current in idx_per_label:
            if idx_current.sum() > 0 :        
                # compute mean
                k_label_mean_sum = h_feature[idx_current].sum(dim=0,keepdim=True) #(1,nfilters)
                k_label_mean     = k_label_mean_sum / idx_current.sum()
                
                # compute cov
                k_delta_cov      = h_feature[idx_current] - k_label_mean   #(nb,nfilters)
                k_label_cov_sum  = (k_delta_cov[:,:,None]*k_delta_cov[:,None,:]).sum(dim=0,keepdim=True)  #(1, nfilters,nfilters)
            else:
                k_label_mean_sum = torch.zeros(1,self.nfeature).to(y.device) 
                k_label_cov_sum  = torch.zeros(1,self.nfeature,self.nfeature).to(y.device)                               
                
            static_mean_sum.append(k_label_mean_sum)     
            static_cov_sum.append(k_label_cov_sum) 
        
        static_mean_sum = torch.cat(static_mean_sum, dim=0)   #(nlables, nfilters)
        static_cov_sum  = torch.cat(static_cov_sum, dim=0)    #(nlables, nfilters, nfilters)
        static_cnt      = torch.tensor([idx_current.sum() for idx_current in idx_per_label]).to(y.device) #(nlables)        
        

        # compute layer feature
        nb = len(y)        
        
        ## update static params per batch
        self.static_cnt_sum.data     +=  static_cnt.data
        self.static_mean_sum.data    +=  static_mean_sum.data
        self.static_cov_sum.data     +=  static_cov_sum.data
        

        return 
    
    def get_static_params(self):

        mean               = self.static_mean_sum.data                          / self.static_cnt_sum.data.view(-1,1)
        mean_shared        = self.static_mean_sum.data.sum(dim=0,keepdims=True) / self.static_cnt_sum.data.sum()
        cov_shared         = self.static_cov_sum.data.sum(dim=0)                / self.static_cnt_sum.data.sum()  
        
        if self.usediagcov:
            cov_shared         = torch.diag(cov_shared) + self.eps_zitter
            precision_shared   = 1 / cov_shared
        else:
            precision_shared   = torch.linalg.inv( cov_shared + self.eps_zitter*torch.eye(self.nfeature).to(mean.device) )
        # return mean,mean_shared,cov_shared,precision_shared
        return mean,mean_shared,precision_shared
        
        

      
    def compute_mh_distance_batch(self, h_feature , datatype = 'x' ,overclass=False, usenormalize=True):
        layers_mean, _ , layers_precision = self.get_static_params()
                           

        ## compute distance                           
        if self.usediagcov :
            delta_h = h_feature[:,None,:] - layers_mean[None,:,:]
            squareterm_inner =  torch.pow(delta_h,2) * layers_precision[None,None,...]  #(nb,ncls,nhdim)*(1,1,nhdim) = (nb,ncls,nhdim)
            mh_distance_over_class  = -0.5*squareterm_inner.sum(dim=-1)       
        else:
            delta_h = h_feature[:,None,:] - layers_mean[None,:,:]
            squareterm_left_product = torch.einsum('bij,bjk->bik',delta_h, layers_precision[None,...].repeat(delta_h.size(0),1,1))
            mh_distance_over_class  = -0.5*(squareterm_left_product*delta_h).sum(dim=-1)

        if usenormalize:
            mh_distance_over_class  = mh_distance_over_class / h_feature.size(-1)
        mh_distance, assigned_label = mh_distance_over_class.max(dim=-1)            

        if overclass:
            return mh_distance_over_class
        else:
            return mh_distance


    def get_nbd_idx(self, h_feature, topk=3):        
        mh_dist_over_cls = self.compute_mh_distance_batch(h_feature , datatype = 'x' ,overclass=True)
        mh_dist,mh_idx = torch.topk(mh_dist_over_cls,k=topk)
        return mh_dist,mh_idx
        
        
        
        
        
        
    # model = FBNN_ResNet_Pro_typeB(BasicBlock, 
    # 184                                 [2,2,2,2],
    # 185                                 in_dim=in_dim, 
    # 186                                 out_dim=out_dim, 
    # 187                                 scalelogstd=configs.model.scalelogstd,
    # 188                                 priorstdout = configs.model.priorstdout,
    # 189                                 temppeak = configs.model.temppeak if 'temppeak' in configs.model.keys() else 0.5,
    # 190     )




class FBNN_ResNet_Pro_typeA(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdin = 1.0, priorstdout = 10., temppeak=1. ):
                
        super(FBNN_ResNet_Pro_typeA,self).__init__()
        

        self.scalelogstd = scalelogstd        
        self.priorstdin  = priorstdin
        self.priorstdout = priorstdout        
        self.temppeak    = temppeak

        
        self.in_planes = 64        
        conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  type='B')
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, type='B')
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, type='B')
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, type='B')
        linear = nn.Linear(512*block.expansion, out_dim)
        mixture_module = Hiddenfeature_Mixture_Module(nfeature=512*block.expansion, nlabel=out_dim, usediagcov=True)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            layer4,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]               
        stochastic_nn    = [
                            linear
                           ]       

        self.deterministic_nn = nn.Sequential(*deterministic_nn)   
        self.stochastic_nn    = nn.Sequential(*stochastic_nn)                       
        self.mixture_module   = mixture_module     
        
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()   
        
        
        
    
         
         

        
        
        

    def _make_layer(self, block, planes, num_blocks, stride, type='A'):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, type))
            self.in_planes = planes * block.expansion            
        return nn.Sequential(*layers)
                 


        
    def _make_layer(self, block, planes, num_blocks, stride, type='A'):        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, type))
            self.in_planes = planes * block.expansion            
        return nn.Sequential(*layers)
                
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean  = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) ##that is solution, will be a solution of MFVI as well.
        return    
    
    
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         


    #def extra_repr(self):
    def __repr__(self):        
        msg =  'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]
        
        
           
           
           
    
    
    # computation checked     
    def compute_funcdist_wo_jacobian(self, x , param_q_mean, param_q_logstd, no_grad_jacobian=False ):
    #def compute_funcdist_wo_jacobian(self, x , param_q_mean, param_q_logstd, no_grad_jacobian=True ):

        #self.setup_functiondist() if not self.flag_setup else 1        
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        h = self.deterministic_nn(x) 
        f_mean = self.func_model(param_q_mean, h ) 

        # use values of jacobian without grad
        if no_grad_jacobian:
            h = h.detach().clone()

        # comopute f_std
        param_q_std  = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov, h
    
    
    
    #def compute_funcdist_wo_jacobian_h(self, h , param_q_mean, param_q_logstd , no_grad_jacobian=False ):
    def compute_funcdist_wo_jacobian_h(self, h , param_q_mean, param_q_logstd , no_grad_jacobian=True ):

        #self.setup_functiondist() if not self.flag_setup else 1        
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        #h = self.deterministic_nn(x) 
        f_mean = self.func_model(param_q_mean, h ) 

        # use values of jacobian without grad
        if no_grad_jacobian:
            h = h.detach().clone()

        # comopute f_std        
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov, h
    

    def build_logit_samples_from_fdist(self, f_mean,f_std,nfsample=1):
        eps = torch.randn(nfsample,*f_mean.shape).to(f_mean.device) #(nfsample,nb,ydlim)
        logits_samples = f_mean[None,...]  + f_std[None,...]*eps
        return logits_samples




    def compute_logit_samples(self, x , y=None, topk=3,  nfsample=1, intrain=True,  use_bayes=True, update_h = False,  configs=None ):                                       
        # set train mode
        self.train() if intrain else self.eval()                                             

        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             

        # compute logit samples            
        qf_mean_x , qf_diagcov_x, h_x  = self.compute_funcdist_wo_jacobian( x, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_x                      = torch.sqrt(qf_diagcov_x)              
                
        
        if not use_bayes:
            nfsample = 1            
            logits_sample = qf_mean_x[None,...] #(1,nb,ydlim)            

            if intrain and update_h and (y is not None):
                self.update_hidden_module_from_hx(h_x, y)                                 

        else:
            nfsample = configs.train.nfsample if configs is not None else nfsample
            logits_sample = self.build_logit_samples_from_fdist(qf_mean_x,qf_std_x,nfsample=nfsample)          
                                    
        return logits_sample



                   

    #def compute_logit_samples(self, x , y=None, use_bayes=True ,intrain=True, nfsample=1, topk=3, configs=None):                               
    def compute_logit_samples_v2(self, x , y=None, topk=3,  nfsample=1, intrain=True,  use_bayes=True, update_h = False,  configs=None):                                       

        # set train mode
        self.train() if intrain else self.eval()                                             

        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             

        # compute logit samples            
        qf_mean_x , qf_diagcov_x, h_x  = self.compute_funcdist_wo_jacobian( x, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_x                      = torch.sqrt(qf_diagcov_x)              
                
        
        if not use_bayes:
            nfsample = 1            
            logits_sample = qf_mean_x[None,...] #(1,nb,ydlim)            
            
            
            if intrain and update_h and (y is not None):
                self.update_hidden_module_from_hx(h_x, y)                                 
                    
            
        else:
            # nfsample = configs.train.nfsample if configs is not None else nfsample
            nfsample = 1                        
            logits_sample = self.build_logit_samples_from_fdist(qf_mean_x,qf_std_x,nfsample=nfsample)          

        return logits_sample, h_x






    def update_hidden_module(self, x, y):
        # update mixture feature
        h = self.deterministic_nn(x)                
        self.mixture_module.update_layers_static_batch(h, y)
        return 
    
    def update_hidden_module_from_hx(self, h_x, y):
        # update mixture feature
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     
    



    def predict_eval(self, x  ,use_bayes=True, topk=3, nfsample=1):                                        
        logits_sample = self.compute_logit_samples(x ,use_bayes=use_bayes ,intrain=False, nfsample=nfsample, topk=topk, configs=None)                                
        pred          = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample   #(nfsample,nb,ydim)      
        return outs       
        
        






    def compute_kde_pfuncdist_from_h(self, ht, param_q_mean, param_q_logstd ,temperature=1e1, requires_grad=False ):
        #import torch.nn.functional as F
        h_mean, _, _       = self.mixture_module.get_static_params()
        h_mean             = h_mean.detach().clone()              #
        kde_ht             = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True, usenormalize=True) ## -0.5*exp()
        kde_weight         = F.softmax( kde_ht , dim=-1)
        proj_ht            = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(1,ncls,ndim) = (nb,ndim)

        f_mean             = self.func_model(param_q_mean, proj_ht)

        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*proj_ht[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
        
        
        
        if len(param_q_logstd) == 2 : #bias term exists
            f_var_bias         = torch.pow( param_q_std[1][None,...], 2)                 
        else:
            f_var_bias         = torch.tensor([0.0]).to(ht.device)

        f_diagcov          = f_var_linear + f_var_bias #(nb,ydim)




        ## revised prior induced by h_mean
        h_mean_square = (param_q_std[0][None,:,:])*(h_mean[:,None,:])    #(1,ncls,nparams) * (ncls,1,nparams)
        h_mean_square = (h_mean_square**2).sum(dim=-1)  #(ncls,ncls)

        phat_idx      = kde_weight.max(dim=-1)[1]
        Mhat          = h_mean_square[phat_idx] #(nb,ncls)

        f_diagcov     = 2*( Mhat + f_var_bias.reshape(1,-1) ) - f_diagcov  #(1,ydim) + (nb,ydim)
        f_diagcov     = f_diagcov.clamp(min=1e-6)
        f_diagcov     = (self.priorstdout**2)*f_diagcov

        
        if requires_grad:        
            # stop gradient
            return f_mean, f_diagcov, proj_ht
        else:
            # stop gradient
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()    
                            
                
    


    def compute_fdist(self, xc , topk=3, useswagparam=True):                

        self.train()                
    
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd              = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        
        qf_mean_xc , qf_diagcov_xc, h_xc  = self.compute_funcdist_wo_jacobian( xc, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_xc                         = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" functional dist on context set
        if not useswagparam:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(xc.device)
            pf_std_xc  = self.priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            
        else:        

            param_p_mean, param_p_logstd   = list(self.params_p_mean) ,  list(self.params_p_logstd) 
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd )      
            pf_mean_xc = pf_mean_xc.detach().clone()
            pf_std_xc  = torch.sqrt(pf_diagcov_xc).detach().clone() 


            if (topk < pf_mean_xc.shape[-1]) :
                nbd_idx    = torch.topk(pf_mean_xc,k = topk,dim=-1)[1]
                qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
                pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    

            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        return param_qf, param_pf, None           
                 

        
    def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
        def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
            kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
            kl_div = kl_div_qf_pf.sum(dim=-1)
            return kl_div #(nb,)
        
        
        # set params
        qf_mean,qf_cov = params_qf
        pf_mean,pf_cov = params_pf

        # split ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
            if weight is None:            
                i_weight_kl_batch = i_kl_batch                
            else:
                i_weight          = weight[i_set]                       
                i_weight_kl_batch = i_weight*i_kl_batch
            kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = (1 / num_batch)*kl_divergence_list.max()
        return torch.clamp(sup_kl,min=eps)    



    
    def get_current_p_params(self):
        return [i_param.data.detach().clone() for i_param in self.stochastic_nn.parameters()]  

           
    def update_params_p(self,param_p_mean,param_p_logstd,base_nan=-12.):

        for i_p_mean,i_p_logstd,i_u_p_mean,i_u_p_logstd in zip(self.params_p_mean,
                                                               self.params_p_logstd,
                                                               param_p_mean,
                                                               param_p_logstd):
            i_p_mean.data=i_u_p_mean.data
            i_p_logstd.data= torch.nan_to_num(i_u_p_logstd.data, nan= base_nan) 
        return             
                     
         
         
                        
    def compute_h_x_grad(self,h_x):
        #print('test h_x_grad')
        h_x_copy = h_x.detach().clone().requires_grad_(True)                                                                                          
        #mh_distance = self.mixture_module.compute_mh_distance_batch(h_x_copy, overcla  ss=True, usenormalize=False)  ## (-0.5*exp()) 
        mh_distance = self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=True, usenormalize=True)  ## (-0.5*exp()) 

        # if version == '1':
        mh_distance_topk1 = torch.topk(mh_distance,k=1)[0]                  
        loss = mh_distance_topk1 - torch.logsumexp(mh_distance,dim=1)
        loss.mean().backward()   

            
        h_x_grad = -h_x_copy.grad.data 
        return h_x_copy, h_x_grad
    
    
                








class FBNN_ResNet_Pro_typeB(FBNN_ResNet_Pro_typeA):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdin = 1.0, priorstdout = 10., temppeak=1. ):


                
        super(FBNN_ResNet_Pro_typeB, self).__init__( block, num_blocks, in_dim, out_dim, scalelogstd, priorstdin, priorstdout, temppeak)        
        
        self.in_planes = 64        
        conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  type='B')
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, type='B')
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, type='B')
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, type='B')
        linear = nn.Linear(512*block.expansion, out_dim)
        mixture_module = Hiddenfeature_Mixture_Module(nfeature=512*block.expansion, nlabel=out_dim, usediagcov=True)

        deterministic_nn = [conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2, 
                            layer3,
                            layer4,
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(start_dim=1),
                           ]               
        stochastic_nn    = [
                            linear
                           ]       

        self.deterministic_nn = nn.Sequential(*deterministic_nn)   
        self.stochastic_nn    = nn.Sequential(*stochastic_nn)                       
        self.mixture_module   = mixture_module     
        
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()
        
        
        






        
        
        
        
        
        