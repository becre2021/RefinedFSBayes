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

eps_pred   = 1e-4
eps_nan    = 1e-4
large_val  = 1e5

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




## --------------------------------------------------------------------------------------------- 
## proposed models
## --------------------------------------------------------------------------------------------- 

import torch
import torch.nn as nn
#from torch.distributions.categorical import Categorical

## module of the penultimate hidden feature 
class Hiddenfeature_Mixture_Module_Regression(nn.Module):     
    def __init__(self, nfeature: int, nlabel: int , usediagcov=False, gamma=0.99,eps_zitter=1e-4):
        super(Hiddenfeature_Mixture_Module_Regression,self).__init__()
        
        self.nlabel = nlabel
        self.nfeature = nfeature
        self.eps_zitter = eps_zitter
        self.gamma = gamma
        self.usediagcov = usediagcov
        
        # per epoch        
        self.cnt               =  nn.Parameter( torch.zeros(1), requires_grad=False )
        self.static_cnt_sum    =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.static_mean_sum   =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature), requires_grad=False )
        self.static_cov_sum    =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature, self.nfeature) , requires_grad=False )        
        
        # set pseudo label
        self.bins_lower        =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.bins_upper        =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )
        self.prepare_bins      =  False
        
        return 

    
    
    def _set_bins(self, y_train, eps=1e-1):
        nbins = self.nlabel

        #if type(y_train) == np.ndarray:
        y_min, y_max = y_train.squeeze().min(), y_train.squeeze().max()
        bins = torch.linspace( y_min - eps, y_max + eps , nbins + 1)
        bins_lower, bins_upper = bins[:nbins], bins[1:nbins+1]

        self.bins_lower.data = torch.tensor(bins_lower).to(y_train.device)
        self.bins_upper.data = torch.tensor(bins_upper).to(y_train.device)

        self.prepare_bins = True
        return 
        
        
        
    def set_pseudolabel(self, y_batch):
        assert self.prepare_bins == True, 'please set bins first'
        
        y_idx = (y_batch >= self.bins_lower)*(y_batch < self.bins_upper) 
        #print('torch.where(y_idx > 0)')
        #print(torch.where(y_idx > 0))
        y_pseudo_label = torch.where(y_idx > 0)[1]
        return y_pseudo_label
        #y_pseudo_label        
        
        
        
    
    
    
    
    
    #def __repr__(self):
    def extra_repr(self):        
        msg = 'nlabel: {}, \n'.format(self.nlabel)
        msg += 'nfeature: {}, \n'.format(self.nfeature)
        return msg[:-1]
    
           
       
    def update_layers_static_batch(self, h_feature, y, eps_zitter=1e-4):
        
        # set pseudob_label 
        y =  self.set_pseudolabel(y)
        
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
        self.cnt.data                +=  1.0
        
        return 
    
    def get_static_params(self):        

        mean               = self.static_mean_sum.data                          / ( self.static_cnt_sum.data.view(-1,1) + eps_nan )
        mean_shared        = self.static_mean_sum.data.sum(dim=0,keepdims=True) / ( self.static_cnt_sum.data.sum()      + eps_nan )
        cov_shared         = self.static_cov_sum.data.sum(dim=0)                / ( self.static_cnt_sum.data.sum()      + eps_nan )
        
        if self.usediagcov:
            cov_shared         = torch.diag(cov_shared) + self.eps_zitter
            precision_shared   = 1 / cov_shared
        else:
            precision_shared   = torch.linalg.inv( cov_shared + self.eps_zitter*torch.eye(self.nfeature).to(mean.device) )
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
            mh_distance_over_class = mh_distance_over_class / h_feature.size(-1)
        mh_distance, assigned_label = mh_distance_over_class.max(dim=-1)            
        
  
        ## if cnt sum == 0       
        list0_idx = [idx_0.item() for idx_0 in torch.where(self.static_cnt_sum == 0)[0]]
        if len(list0_idx) > 0:
            # if len(mh_distance.data) < 20:
            #     breakpoint()            
            
            for ith_idx in list0_idx:
                mh_distance_over_class.data[:,ith_idx] = -large_val*torch.ones_like(mh_distance_over_class[:,ith_idx]).to(layers_mean.device)
                #mh_distance.data[ith_idx]              = -large_val*torch.ones_like(mh_distance[ith_idx]).to(layers_mean.device)      

            mh_distance = mh_distance_over_class.max(dim=-1)[0]
            #breakpoint()
        
        

        if overclass:
            return mh_distance_over_class
        else:
            return mh_distance




        
        
        

## use context set as adversarial hidden feature
class FBNN_SmallNet_Pro_typeA(nn.Module):

    def __init__(self, in_dim=3, out_dim=1, hidden_dim=50, nbins=20, scalelogstd = -8, noiselogstd= 1. , priorstdin = 1.0, priorstdout = 10. , temppeak=1.):

        super(FBNN_SmallNet_Pro_typeA, self).__init__()


        self.scalelogstd          = scalelogstd        
        self.priorstdin           = priorstdin
        self.priorstdout          = priorstdout       

        self.temperature_distance = 1e0
        self.temperature_peak     = temppeak
        self.temperature_soft     = 1e0 
        
        
        

    
        ## pseudo label as out_dim
        mixture_module = Hiddenfeature_Mixture_Module_Regression(nfeature=hidden_dim, nlabel=nbins, usediagcov=True)
        self.mixture_module   = mixture_module     

        #hidden_dim = 50
        layer1 = nn.Linear(in_dim, hidden_dim)
        layer2 = nn.Linear(hidden_dim, out_dim)

        deterministic_nn = [layer1,nn.ReLU(),
                           ]       
        
        stochastic_nn    = [layer2,
                           ]       
            

        self.deterministic_nn = nn.Sequential(*deterministic_nn)   
        self.stochastic_nn    = nn.Sequential(*stochastic_nn)                       
        self.noise_logstd     = nn.Parameter(torch.tensor(np.log(noiselogstd + eps_log))  , requires_grad=True )    

        self.deterministic_nn.apply(_weights_init)          
        self.stochastic_nn.apply(_weights_init)        

        self.setup_parameters()
        self.setup_functiondist()
        

                
    def setup_parameters(self):        
        self.num_params_d    = _num_parameters(self.deterministic_nn)          
        self.num_params_s    = _num_parameters(self.stochastic_nn)  

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
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
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


       
       

    def compute_logit_samples(self, x , nfsample = 1, use_bayes=True):                                       

        # set train mode
        self.eval()                                             

        # get params
        param_q_mean, param_q_logstd   = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             

        # compute logit samples            
        qf_mean_x , qf_diagcov_x, _    = self.compute_funcdist_wo_jacobian( x, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_x                       = torch.sqrt(qf_diagcov_x)                              

        # compute noise params
        f_noise   = torch.exp(2*self.noise_logstd)*torch.ones_like(qf_mean_x).to(qf_mean_x.device)
                                                                                        
        
        if not use_bayes:          
            qf_mean_x = qf_mean_x[None,...]
            f_noise   = f_noise[None,...]              
            return qf_mean_x, f_noise
                                
        else:
            f_samples = self.build_logit_samples_from_fdist(qf_mean_x, qf_std_x, nfsample=nfsample)  
            f_noise   =  f_noise[None,...].repeat(nfsample,1,1)                      
            return f_samples, f_noise
                                            
                                            

    ## inference using the adversarial hidden features 
    def compute_logit_samples_v2(self, x , y=None, intrain=True,  use_bayes=True, update_h = False,  configs=None):                                       

        # set train mode
        self.train() if intrain else self.eval()                                             

        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             

        # compute logit samples            
        qf_mean_x , qf_diagcov_x, h_x  = self.compute_funcdist_wo_jacobian( x, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_x                      = torch.sqrt(qf_diagcov_x)              
                
        # compute noise params        
        f_noise   = torch.exp(2*self.noise_logstd)*torch.ones_like(qf_mean_x).to(qf_mean_x.device)
        
        if not use_bayes:            
            ## update the parameters of the empirical hidden feature 
            if intrain and update_h and (y is not None):
                self.update_hidden_module_from_hx(h_x, y)                                 

            qf_mean_x = qf_mean_x[None,...]
            f_noise   = f_noise[None,...]

            return qf_mean_x, f_noise, h_x
                                
        else:
            nfsample  = configs.train.nfsample
            # # return logits_sample, h_x
            f_samples = self.build_logit_samples_from_fdist(qf_mean_x, qf_std_x, nfsample=nfsample)            
            f_noise   =  f_noise[None,...].repeat(nfsample,1,1)

            return f_samples, f_noise, h_x
        
               
       
       
       
    ## update mixture feature during training
    def update_hidden_module(self, x, y):
        h = self.deterministic_nn(x)                
        self.mixture_module.update_layers_static_batch(h, y)
        return        
                
    ## update mixture feature during training
    def update_hidden_module_from_hx(self, h_x, y):
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     
       
                   
           
    ##  ICML : modified version for construcing prior  
    def compute_kde_pfuncdist_from_h(self, ht, param_q_mean, param_q_logstd ,requires_grad=False, temperature=1e1 ):
        #import torch.nn.functional as F
        h_mean, _, _ = self.mixture_module.get_static_params()
        
        #breakpoint()

        kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
        kde_weight = F.softmax( kde_ht, dim=-1)
        proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

        f_mean = self.func_model(param_q_mean, proj_ht)
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*proj_ht[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
                
        if len(param_q_logstd) == 2 : #bias term exists
            f_var_bias         = torch.pow( param_q_std[1][None,...], 2)                 
        else:
            f_var_bias         = torch.tensor([0.0]).to(ht.device)

        f_diagcov          = f_var_linear + f_var_bias #(nb,ydim)


        ## build informative prior 
        h_mean_square = (param_q_std[0])*(h_mean)                  #(ydim,nparams) * (ncls,ndim)
        h_mean_square = torch.pow(h_mean_square,2).sum(dim=-1)  #(ydim,1)
        f_diagcov = 2*( h_mean_square.reshape(1,-1) + f_var_bias.reshape(1,-1) ) - f_diagcov  #(1,ydim) + (nb,ydim)
        f_diagcov = f_diagcov.clamp(min=1e-12)
        f_diagcov = (self.priorstdout**2)*f_diagcov        
        
        
        ## projectino step for regression
        idx_estimate = kde_weight.max(dim=1)[1]        
        f_diagcov = torch.tensor([ith_cov[ith_entry].data for ith_cov,ith_entry in zip(f_diagcov,idx_estimate)]).to(ht.device)        
        f_diagcov = f_diagcov.reshape(-1,1)
        
        #breakpoint()
        

                
        # stop gradient
        if requires_grad:
            return f_mean, f_diagcov, proj_ht
        else:
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()

    
    
    
    def compute_fdist(self, xc , topk=3, useswagparam=True):                

        self.train()                
    
        # compute "q" function space dist on context inputs        
        param_q_mean, param_q_logstd      = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)             
        qf_mean_xc , qf_diagcov_xc, h_xc  = self.compute_funcdist_wo_jacobian( xc, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_xc                         = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" function space dist on context inputs        
        if not useswagparam:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(xc.device)
            pf_std_xc  = self.priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            
        else:        

            ## compute "BMA" function space dist on context inputs        
            param_p_mean, param_p_logstd  = list(self.params_p_mean) ,  list(self.params_p_logstd) 
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd )      
            pf_mean_xc = pf_mean_xc.detach().clone()
            pf_std_xc  = self.priorstdout*torch.sqrt(pf_diagcov_xc).detach().clone()


            
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


    ## get the last layer weight parameters
    def get_current_p_params(self):
        return [i_param.data.detach().clone() for i_param in self.stochastic_nn.parameters()]  
           
    ## update the last layer weight parameters
    def update_params_p(self,param_p_mean,param_p_logstd,base_nan=-12.):

        for i_p_mean,i_p_logstd,i_u_p_mean,i_u_p_logstd in zip(self.params_p_mean,
                                                               self.params_p_logstd,
                                                               param_p_mean,
                                                               param_p_logstd):
            i_p_mean.data=i_u_p_mean.data
            i_p_logstd.data= torch.nan_to_num(i_u_p_logstd.data, nan= base_nan) 
        return             
                
    


    
    ##-------------------------------------------------------------
    ## build the adversional context hidden feature
    ##-------------------------------------------------------------
    
    def compute_h_x_grad(self,h_x):
        h_x_copy = h_x.detach().clone().requires_grad_(True)                                                                                          
        #mh_distance = -self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=False, usenormalize=False)  
        mh_distance = -self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=False, usenormalize=True)  
        mh_distance.mean().backward()   
        h_x_grad = h_x_copy.grad.data 
        return h_x_copy, h_x_grad
            
            
    def compute_fdist_from_h(self, h_xc , topk=3, useswagparam=True):                
        self.train()                
    
        # compute "q" function space dist on context inputs        
        param_q_mean, param_q_logstd    = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        qf_mean_xc , qf_diagcov_xc, _   = self.compute_funcdist_wo_jacobian_h( h_xc, param_q_mean, param_q_logstd )          ## v20 revised        
        qf_std_xc                       = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" function space dist on context inputs        
        if not useswagparam:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(h_xc.device)
            pf_std_xc  = self.priorstdout*torch.ones_like(qf_mean_xc).to(h_xc.device)            
        else:        
            #breakpoint()
            ## compute "BMA" function space dist on context inputs                    
            param_p_mean, param_p_logstd   = list(self.params_p_mean) ,  list(self.params_p_logstd) 
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd, requires_grad=False )      
            pf_mean_xc = pf_mean_xc.detach().clone()
            pf_std_xc  = self.priorstdout*torch.sqrt(pf_diagcov_xc).detach().clone()


            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)


        return param_qf, param_pf, None       
    
    
    
    #def predict_eval(self, x  ,use_bayes=True):                                        
    def predict_eval(self, x  , nfsample=1, use_bayes=True):                                        

        f_samples, f_noise = self.compute_logit_samples(x ,nfsample=nfsample, use_bayes=use_bayes)                                
                                
        outs = AttrDict()
        outs.f_mean = f_samples  #(nfsample,nb,nout)        
        outs.f_std  = f_noise    #(nfsample,nb,nout)      
    
        return outs       
        
                
        
        
        
        
        
        