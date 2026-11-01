import random
import numpy as np
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils import parameters_to_vector
from torch.autograd import Variable
        
from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers

    

from models_methodology import build_MVN, compute_kl
from models_methodology import Model_Base    


# from models_bayesian_layers import (BayesLinear_localreparam, 
#                                    BayesConv2d_localreparam
# )

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from attrdict import AttrDict


from torch.distributions.kl import kl_divergence


from functorch import vmap,grad,jacfwd,jacrev


from models_methodology import build_MVN
from torch.distributions.kl import kl_divergence


__all__ = ['model_deep_list','_setup_model_deep'
           ]



## ---------------------------------------------------------------------        
## setup deep model
## ---------------------------------------------------------------------            

model_deep_list = ['resnet18','resnet50',                      
                   'bmafbresnet18','bmafbresnet50',
                   'bmafbresnet18st','bmafbresnet50st',
                  ]

def _setup_model_deep(configs: AttrDict):
    if configs.model.modelname not in model_deep_list:
        raise ValueError(f"Unsupported model {configs.model.modelname!r}; choose from {model_deep_list}")
    in_dim, out_dim = configs.model.in_dim, configs.model.out_dim
    hidden_dim = 128 if configs.model.hidden_dim is None else configs.model.hidden_dim



    if configs.model.modelname in ['resnet18']:
        model = ResNet_typeB(BasicBlock, [2,2,2,2], in_dim=in_dim, out_dim=out_dim)         


    if configs.model.modelname in ['resnet50'] :        
        model = ResNet_typeB(Bottleneck, [3,4,6,3], in_dim=in_dim, out_dim=out_dim)        


    if configs.model.modelname == 'bmafbresnet18':  
        model = FBNN_ResNet_Pro_typeB(BasicBlock, 
                                    [2,2,2,2],
                                    in_dim=in_dim, 
                                    out_dim=out_dim, 
                                    scalelogstd=configs.model.scalelogstd,
                                    priorstdout = configs.model.priorstdout,
        )


    if configs.model.modelname == 'bmafbresnet50': 
        model = FBNN_ResNet_Pro_typeB(Bottleneck, 
                                        [3,4,6,3], 
                                        in_dim=in_dim, 
                                        out_dim=out_dim, 
                                        scalelogstd=configs.model.scalelogstd,
                                        priorstdout = configs.model.priorstdout,
        )                        


    if configs.model.modelname == 'bmafbresnet18st': 
        model = FBNN_ResNet_Pro_typeC(BasicBlock_ST, 
                                    [2,2,2,2],
                                    in_dim=in_dim, 
                                    out_dim=out_dim, 
                                    scalelogstd=configs.model.scalelogstd,
                                    priorstdout = configs.model.priorstdout,
        )                        


    if configs.model.modelname == 'bmafbresnet50st': 
        model = FBNN_ResNet_Pro_typeC(Bottleneck_ST, 
                                        [3,4,6,3], 
                                        in_dim=in_dim, 
                                        out_dim=out_dim, 
                                        scalelogstd=configs.model.scalelogstd,
                                        priorstdout = configs.model.priorstdout,
        )                        



    return model 










def _num_parameters(model):
    return sum([p.numel() for p in model.parameters()])

## --------------------------------------------------------------------------------------------- 
## init weights
## ---------------------------------------------------------------------------------------------- 

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


def transform_logstd(logstd):
    return F.softplus(logstd, beta=1, threshold=20)

##-----------------------------------------------------------------------------------------------
## Bathcnorm revision
##-----------------------------------------------------------------------------------------------
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









## --------------------------------------------------------------------------------------------- 
## resnet block
##  reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py         for resnet 20,32,44,56,110,1202 
##             https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py            for resnet 18,34,50,101,152 
## ---------------------------------------------------------------------------------------------- 
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
    
    
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.option = option


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                
            if option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out    



from models_bayeslayers import Conv2d_VI    

class BasicBlock_ST(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1,   option='A', scalelogstd=-6):
        super(BasicBlock_ST, self).__init__()
        
        self.scalelogstd = scalelogstd
        self.conv1 = Conv2d_VI(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,    scalelogstd=scalelogstd)
        self.conv2 = Conv2d_VI(planes, planes,    kernel_size=3, stride=1,     padding=1,  bias=False ,   scalelogstd=scalelogstd)
        self.bn1   = BatchNorm2d_v2(planes)  # self.bn1 = nn.BatchNorm2d(planes , track_running_stats=True)
        self.bn2   = BatchNorm2d_v2(planes)  # self.bn2 = nn.BatchNorm2d(planes , track_running_stats=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d_VI(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,  scalelogstd=scalelogstd),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        #print(f'scalelogstd={scalelogstd} initialized')  ##init checked
                

    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out
    
    

class Bottleneck_ST(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1 ,  option='B' , scalelogstd=-6):
        super(Bottleneck_ST, self).__init__()
        
        self.scalelogstd = scalelogstd
        
        self.conv1 = Conv2d_VI(in_planes, planes, kernel_size=1, bias=False , scalelogstd=scalelogstd)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_VI(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False , scalelogstd=scalelogstd)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_VI(planes, self.expansion *planes, kernel_size=1, bias=False , scalelogstd=scalelogstd)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)
        self.option = option
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                
            if option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d_VI(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False  , scalelogstd=scalelogstd),
                     nn.BatchNorm2d(self.expansion * planes)
                )


        #print(f'scalelogstd={scalelogstd} initialized')  ##init checked


    def forward(self, x ):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out    
                      
    


## --------------------------------------------------------------------------------------------- 
## resnet type A for 20,32,44,56,110,1202 : use the smaller number of parameters
## resnet type B for 18,34,50,101,152  
## resnet 20: .27M parameters, resnet 18:  11.4M parameters
## ---------------------------------------------------------------------------------------------- 

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
    
        
    #def predict_eval(self, x , nfsample=1):        
    def predict_eval(self, x , nfsample=1, use_bayes=None):        
        self.eval()        
        y = self.deterministic_nn(x)
        pred = F.softmax(y,dim=-1)           
        
        outs = AttrDict()
        outs.pred = pred  #prob               
        outs.logits = y.unsqueeze(dim=0)
        return outs            
    


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
            

    #def predict_eval(self, x , nfsample=1):        
    def predict_eval(self, x , nfsample=1, use_bayes=None):        
        self.eval()        
        y = self.deterministic_nn(x)
        pred = F.softmax(y,dim=-1)           
        
        outs = AttrDict()
        outs.pred = pred  #prob               
        outs.logits = y.unsqueeze(dim=0)
        return outs            
    





  
  
    
## ------------------------------------------------------------------------------------------------------------------------------------------         
## Bayesian ResNET CNN        
## ------------------------------------------------------------------------------------------------------------------------------------------         

        
    
    

        
    
            
            
## --------------------------------------------------------------------------------------------- 
## BNN LastLayers Resnet
## ---------------------------------------------------------------------------------------------- 
    
    
        
        
























## --------------------------------------------------------------------------------------------- 
## FBNN Resnet Baseline structure
## ---------------------------------------------------------------------------------------------- 



    
        
    
           
           
    
        
        
        
        
        
        
        

           

            
                
 



##  --------------------------------------------------------------------------------------------------------------------------  ##
## FBNN Resnet proposed 
##  --------------------------------------------------------------------------------------------------------------------------  ##

#class FBSPNN4_ResNet(FBNN_Base):


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




## use diag cov
class Hiddenfeature_Mixture_Module(nn.Module):     
    def __init__(self, nfeature: int, nlabel: int , usediagcov=True, gamma=0.99,eps_zitter=1e-4):
        super(Hiddenfeature_Mixture_Module,self).__init__()
        
        self.nfeature = nfeature
        self.nlabel = nlabel
        self.eps_zitter = eps_zitter
        #self.gamma = gamma
        self.usediagcov = usediagcov
        
        # per epoch        
        self.static_cnt_sum    =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.static_mean_sum   =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature), requires_grad=False )
        self.static_cov_sum    =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature) , requires_grad=False )        
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
                k_delta_cov      = h_feature[idx_current] - k_label_mean    #(nb,nfilters)
                k_label_cov_sum  = (k_delta_cov**2).sum(dim=0,keepdim=True)  #(1,nfilters)
            else:
                k_label_mean_sum = torch.zeros(1,self.nfeature).to(y.device) 
                k_label_cov_sum  = torch.zeros(1,self.nfeature).to(y.device)                               
                
            static_mean_sum.append(k_label_mean_sum)     
            static_cov_sum.append(k_label_cov_sum) 
        
        static_mean_sum = torch.cat(static_mean_sum, dim=0)   #(nlables, nfilters)
        static_cov_sum  = torch.cat(static_cov_sum, dim=0)    #(nlables, nfilters)
        static_cnt      = torch.tensor([idx_current.sum() for idx_current in idx_per_label]).to(y.device) #(nlables)        
        

        # compute layer feature
        nb = len(y)        
        
        # update static params per batch
        self.static_cnt_sum.data     +=  static_cnt.data
        self.static_mean_sum.data    +=  static_mean_sum.data
        self.static_cov_sum.data     +=  static_cov_sum.data
        

        return 
    
    def get_static_params(self):

        mean               = self.static_mean_sum.data                          / self.static_cnt_sum.data.view(-1,1)
        mean_shared        = self.static_mean_sum.data.sum(dim=0,keepdims=True) / self.static_cnt_sum.data.sum()
        cov_shared         = self.static_cov_sum.data.sum(dim=0)                / self.static_cnt_sum.data.sum()  
        
        
        # if self.usediagcov:
        cov_shared         = cov_shared + self.eps_zitter
        precision_shared   = 1 / cov_shared 

        return mean,mean_shared,precision_shared
        

    def compute_mh_distance_batch(self, h_feature  ,overclass=False, usenormalize=True):    
        layers_mean, _ , layers_precision = self.get_static_params()
                           

        ## compute distance                           
        # if self.usediagcov :
        delta_h = h_feature[:,None,:] - layers_mean[None,:,:]
        squareterm_inner =  torch.pow(delta_h,2) * layers_precision[None,None,...]  #(nb,ncls,nhdim)*(1,1,nhdim) = (nb,ncls,nhdim)
        mh_distance_over_class  = -1*squareterm_inner.sum(dim=-1)       

        if usenormalize:
            mh_distance_over_class  = mh_distance_over_class / h_feature.size(-1)
        mh_distance, assigned_label = mh_distance_over_class.max(dim=-1)            

        if overclass:
            return mh_distance_over_class
        else:
            return mh_distance
        



    
        
usenormalize=True        

class FBNN_ResNet_Pro_typeA(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdout = 10.):

        self.scalelogstd          = scalelogstd        
        self.priorstdout          = priorstdout       


        self.temperature_distance = 1 ## original
        #self.temperature_distance = 1e-2 


        
        
        
        super(FBNN_ResNet_Pro_typeA, self).__init__()
        self.in_planes = 16        
        conv1 = nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)        
        bn1 = BatchNorm2d_v2(16) 
        layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)                
        linear = nn.Linear(64, out_dim)
        mixture_module = Hiddenfeature_Mixture_Module(nfeature=64, nlabel=out_dim, usediagcov=True)

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
                
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        self.params_p_mean  = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) ##that is solution, will be a solution of MFVI as well.
        return    
    
    
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         


    def __repr__(self):        
        msg =  'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]
        
        
           
    def compute_funcdist_wo_jacobian(self, x , param_q_mean, param_q_logstd, no_grad_jacobian=False ):

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





               

    def compute_logit_samples(self, x , y=None,   nfsample=1, intrain=True,  use_bayes=True, update_h = False,  configs=None):                                       

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


        
                               
    def compute_kde_pfuncdist_from_h(self, ht, param_p_mean, param_p_logstd, requires_grad=False):
        h_mean, _, _       = self.mixture_module.get_static_params()
        h_mean             = h_mean.detach().clone()              #
        kde_ht             = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True, usenormalize=True) 
        kde_weight         = F.softmax( kde_ht , dim=-1)
        proj_ht            = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(1,ncls,ndim) = (nb,ndim)


    
        m_qx               = h_mean[kde_weight.argmax(dim=-1)]
        f_mean             = self.func_model(param_p_mean, proj_ht)



        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd ]                
        # param_q_std = [10 * transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd]  # Outer variance factor = 100 (including bias)
        f_var_linear_inner = self.priorstdout*(param_q_std[0][None,...])*((m_qx - proj_ht)[:,None,:]) #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
        f_var_linear      += torch.pow((param_q_std[0][None,...])*(m_qx[:,None,:]),2).sum(dim=-1)
        
            
        
        if len(param_p_logstd) == 2 : #bias term exists
            f_var_bias         = torch.pow( param_q_std[1][None,...], 2)                 
        else:
            f_var_bias         = torch.tensor([0.0]).to(ht.device)

        f_diagcov          = f_var_linear + f_var_bias #(nb,ydim)


        if requires_grad:        
            # stop gradient
            return f_mean, f_diagcov, proj_ht
        else:
            # stop gradient
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()                                        
                                                                      
    
    
    def compute_fdist(self, xc , useswagparam=True):                

        self.train()                
    
        param_q_mean, param_q_logstd              = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        
        #qf_mean_xc , qf_diagcov_xc, h_feature_xc  = self.compute_funcdist_wt_jacobian( xc, param_q_mean, param_q_logstd )  #with jacobian computation          
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



            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        return param_qf, param_pf, None                    
           
               


    def compute_fdist_from_h(self, h_xc ,  useswagparam=True):                
        self.train()                
    
        param_q_mean, param_q_logstd    = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        qf_mean_xc , qf_diagcov_xc, _   = self.compute_funcdist_wo_jacobian_h( h_xc, param_q_mean, param_q_logstd )               
        
        qf_std_xc                       = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" functional dist on context set
        if not useswagparam:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(h_xc.device)
            pf_std_xc  = self.priorstdout*torch.ones_like(qf_mean_xc).to(h_xc.device)            
        else:        
            param_p_mean, param_p_logstd  = list(self.params_p_mean) ,  list(self.params_p_logstd) 
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd, requires_grad=False )      
            pf_mean_xc = pf_mean_xc.detach().clone()
            pf_std_xc  = torch.sqrt(pf_diagcov_xc).detach().clone() 



            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        #breakpoint()
        #return param_qf, param_pf 
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
            i_weight_kl_batch = i_kl_batch                
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
            




    def predict_eval(self, x  ,use_bayes=True,  nfsample=1):                                        

        # set train mode
        self.eval()                                             

        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             

        # compute logit samples            
        qf_mean_x , qf_diagcov_x, h_x  = self.compute_funcdist_wo_jacobian( x, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_x                      = torch.sqrt(qf_diagcov_x)              
                
        
        if not use_bayes:
            nfsample = 1            
            logits_sample = qf_mean_x[None,...] #(1,nb,ydlim)            

        else:
            logits_sample = self.build_logit_samples_from_fdist(qf_mean_x,qf_std_x,nfsample=nfsample)      
        
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout)                                 
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample   #(nfsample,nb,ydim)      
        return outs       
        
        


    def compute_prior_samples(self,x, nfsample=1):
        
        h_xc = self.deterministic_nn(x) 

        ## kde + swag diag +             
        param_p_mean, param_p_logstd   = list(self.params_p_mean) ,  list(self.params_p_logstd) 
        pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd, requires_grad=False )      
        pf_mean_xc = pf_mean_xc.detach().clone()
        pf_std_xc  = torch.sqrt(pf_diagcov_xc).detach().clone()
        f_samples  = self.build_logit_samples_from_fdist(pf_mean_xc,pf_std_xc,nfsample=nfsample)

        return f_samples

    
    
    ##--------------------
    ## build h_x
    ##--------------------
    

    def update_hidden_module(self, x, y):
        # update mixture feature
        h = self.deterministic_nn(x)                
        self.mixture_module.update_layers_static_batch(h, y)
        return 
    
    def update_hidden_module_from_hx(self, h_x, y):
        # update mixture feature
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     
    


    def compute_h_x_grad(self,h_x):
        h_x_copy = h_x.detach().clone().requires_grad_(True)                                                                                          
        mh_distance = self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=True, usenormalize=True)  ## (-0.5*exp()) 

        mh_distance_topk1 = torch.topk(mh_distance,k=1)[0]                  
        loss              = mh_distance_topk1 - torch.logsumexp(mh_distance,dim=1)
        loss.mean().backward()   
        h_x_grad = -h_x_copy.grad.data 
        return h_x_copy, h_x_grad

            
                        
                        
                    
                    
                    
                    
        
class FBNN_ResNet_Pro_typeB(FBNN_ResNet_Pro_typeA):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdout = 10.):


                
        super(FBNN_ResNet_Pro_typeB, self).__init__( block, num_blocks, in_dim, out_dim, scalelogstd, priorstdout)        
        
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
        self.stochastic_nn    =  nn.Sequential(*stochastic_nn)                       
        self.mixture_module   = mixture_module     
        
        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()
        
    






      
      

class FBNN_ResNet_Pro_typeC(FBNN_ResNet_Pro_typeB):

    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, priorstdout = 10.):
                
        super(FBNN_ResNet_Pro_typeC, self).__init__( block, num_blocks, in_dim, out_dim, scalelogstd, priorstdout)        
        
        self.in_planes = 64        
        self.scalelogstd = scalelogstd

        conv1 = Conv2d_VI(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False,   scalelogstd=self.scalelogstd)
        
        
        bn1    = nn.BatchNorm2d(64)
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
            layers.append(block(self.in_planes, planes, stride, type ,  self.scalelogstd))
            self.in_planes = planes * block.expansion            
        return nn.Sequential(*layers)
    
    

    # computation checked     
    def compute_funcdist_wo_jacobian(self, x , param_q_mean, param_q_logstd, no_grad_jacobian=False ):

        assert len(param_q_logstd) <=2
        
        # compute f_mean
        h      = self.deterministic_nn(x) 
        f_mean = self.stochastic_nn(h)

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
    
    
    

    def compute_logit_samples(self, x , y=None,   nfsample=1, intrain=True,  use_bayes=True, update_h = False,  configs=None):                                       

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
    
    
            
    
    

    def predict_eval(self, x  ,use_bayes=True,  nfsample=1):               
        
        
        per_sample = 1
        ntrial     = nfsample // per_sample


        if use_bayes:        
            logits_sample = []
            for _ in range(ntrial):                                 
                i_logits_sample, _ = self.compute_logit_samples(x ,use_bayes=use_bayes ,intrain=False, nfsample=per_sample,  configs=None) 
                logits_sample.append( i_logits_sample )                           
            logits_sample = torch.cat(logits_sample,dim=0)        
                        

        else:
            logits_sample, _ = self.compute_logit_samples(x ,use_bayes=use_bayes ,intrain=False, nfsample=1, configs=None)

        
        
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample   #(nfsample,nb,ydim)      
        return outs       
        
        



    def compute_prior_samples(self,x, nfsample=1):

        f_samples_list = []        
        for _ in range(nfsample):
            h_xc = self.deterministic_nn(x) 

            ## kde + swag diag +             
            param_p_mean, param_p_logstd   = list(self.params_p_mean) ,  list(self.params_p_logstd) 
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd, requires_grad=False )      
            pf_mean_xc = pf_mean_xc.detach().clone()
            pf_std_xc  = torch.sqrt(pf_diagcov_xc).detach().clone()  
            f_samples  = self.build_logit_samples_from_fdist(pf_mean_xc,pf_std_xc,nfsample=1)
            f_samples_list.append(f_samples)

        return torch.cat(f_samples_list,dim=0)

        
        
        
        
        
        
        
        
        