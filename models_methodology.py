import random
import numpy as np
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F

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


## ---------------------------------------------------------------------------------------------------------
## base utility 
##---------------------------------------------------------------------------------------------------------

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




## ---------------------------------------------------------------------------------------------------------
## base structure
##---------------------------------------------------------------------------------------------------------
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
        #sup_kl = torch.tensor(0.0).to(x.device)                    
        param_vec = parameters_to_vector(self.parameters())
        reg = 0.5 * torch.norm(param_vec, p = 2)
        outs = AttrDict()
        outs.logits = logits
        outs.reg = reg                
        return outs        
        
        
    def predict_eval(self, x , nfsample=1, topk=1, use_bayes=False):        
        self.eval()        
        y = self.deterministic_nn(x)
        pred = F.softmax(y,dim=-1)           
        
        outs = AttrDict()
        outs.pred = pred  #prob               
        outs.logits = y.unsqueeze(dim=0)
        return outs            




##---------------------------------------------------------------------------------------------------------
##  BNN_MFVI_BASE Revistied
##---------------------------------------------------------------------------------------------------------
class BNN_Base_Revisited(nn.Module):    
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(BNN_Base_Revisited,self).__init__()
    
        #self.prestochastic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

    #def extra_repr(self):
    def __repr__(self):        
        msg = 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#stochastic qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd_s))
        msg += '#stochastic pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_s),_num_parameters(self.params_p_logstd_s))
        return msg[:-1]

        
    def setup_parameters(self):        
        self.num_params_s      = _num_parameters(self.stochastic_nn)  
        self.params_q_logstd_s =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
        self.params_p_mean_s   = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd_s = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        
        return    
    
    
    def setup_functiondist(self):    
        func_model_s, _ ,  buffers_s = make_functional_with_buffers(self.stochastic_nn)                
        self.func_model_s            =   lambda params, x_in : func_model_s(params, buffers_s, x_in) 
        
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
                
        func_samples = []
        for _ in range(nfsample):
            ith_weight_sample_s = self.sample_weight(param_q_mean_s, param_q_logstd_s)                        
            ith_func_samples = self.func_model_s(ith_weight_sample_s , x)                            
            func_samples.append(  ith_func_samples[None,...]  )  

        func_samples = torch.cat(func_samples,dim=0) #(nfsample,nb,ydim)           
        
        return func_samples

                
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            #nfsample = configs.train.nfsample
            nfsample = configs.train.nfsample
            
        else:            
            nctset = 2 
            reglambda = 1e-3
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
            weight_kl = self.compute_weight_kl(params_qf, params_pf)

        else :
            weight_kl = torch.tensor(0.0).to(x.device)            
        #reg_kl = reglambda*sup_kl
        
            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = weight_kl               
   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1, use_bayes=False):        
        self.eval()        
        # get variational params    
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x, param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
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
    
    

##---------------------------------------------------------------------------------------------------------
## Tractable Functional BNN BASE
##---------------------------------------------------------------------------------------------------------
class FBNN_Base(nn.Module):
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(FBNN_Base,self).__init__()
    
        self.deterministic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

    #def extra_repr(self):
    def __repr__(self):        
        msg = 'deterministic NN: {} \n'.format(self.deterministic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]

        
    def setup_parameters(self):        
        self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        #self.params_logstd = nn.Parameter(torch.empty(self.num_params_s).uniform_(-5,-4))    
        #self.params_q_logstd = nn.Parameter(torch.empty(self.num_params_s).uniform_(self.scalelogstd,self.scalelogstd+1) )    
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        #self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )

        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,0+1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(-1,0)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) ##failed
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()])  #3failed
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()])  #3failed

        return    
    
    
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         


    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                
            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)
    
        
    def sample_weight(self, param_q_mean, param_q_logstd):        
        weight_sample = []
        for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
            ith_weight_sample = ith_q_mean + transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)                       
            weight_sample.append(ith_weight_sample)
        return weight_sample
        
                        
    def compute_func_samples(self, x, param_q_mean, param_q_logstd, nfsample=5):                
        self.setup_functiondist() if not self.flag_setup else 1
        
        h = self.deterministic_nn(x)        
        func_samples = []
        for _ in range(nfsample):
            ith_weight_sample = self.sample_weight(param_q_mean, param_q_logstd)            
            func_samples.append(  self.func_model(ith_weight_sample , h)[None,...]  )  
        func_samples = torch.cat(func_samples,dim=0) #(nfsample,nb,ydim)           
        return func_samples


    def compute_func_dist(self, x , param_q_mean, param_q_logstd ):
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
        return f_mean, f_cov
                
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            #nfsample = configs.train.nfsample
            nfsample = 1            
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()            
        
        # get params
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute function samples
        #logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)
        
        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
            # set functional dist on x u xc                            
            pf_mean_x ,  pf_cov_x = self.compute_func_dist( x, param_p_mean, param_p_logstd )            
            qf_mean_x ,  qf_cov_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
            
            pf_mean_xc , pf_cov_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
            qf_mean_xc , qf_cov_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

            pf_mean    , pf_cov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_cov_x,pf_cov_xc],dim=0)            
            qf_mean    , qf_cov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_cov_x,qf_cov_xc],dim=0)
        
        
            #print('pf_mean_x: {}, qf_mean_x: {}'.format(pf_mean_x,qf_mean_x))
            #print('pf_cov_x[0]: {}, qf_cov_x[0]: {}'.format(pf_cov_x[0],qf_cov_x[0]))
        
        
            # compute kl
            sup_kl = self.compute_sup_kl([qf_mean,qf_cov], [pf_mean,pf_cov], Lowrank=False, nctset=nctset) 
        else :
            sup_kl = torch.tensor(0.0).to(x.device)            
        #reg_kl = reglambda*sup_kl
        
            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = sup_kl                
   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1, use_bayes=False):        
        self.eval()        
        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()),  list(self.params_q_logstd)     
        
        # compute function samples
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        return outs       



    def compute_sup_kl(self, params_qf, params_pf, Lowrank=True, nctset=2, eps=1e-8):
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
            i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank )        
            kl_divergence_list.append(i_kl)

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        return torch.clamp(kl_divergence_list.max(),min=eps)    
    
    
    
    
    
    
    
    
    
    
##---------------------------------------------------------------------------------------------------------
## FBNN_BASE Revistied
##---------------------------------------------------------------------------------------------------------
class FBNN_Base_Revisited(nn.Module):    
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(FBNN_Base_Revisited,self).__init__()
    
        self.prestochastic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

    #def extra_repr(self):
    def __repr__(self):        
        msg = 'prestohcastic NN: {} \n'.format(self.prestochastic_nn)
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#pre stochastic qmu: {}, #qstd: {} \n'.format(_num_parameters(self.prestochastic_nn),_num_parameters(self.params_q_logstd_ps))        
        msg += '#pre stochastic pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_ps),_num_parameters(self.params_p_logstd_ps))              
        msg += '#stochastic qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd_s))
        msg += '#stochastic pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_s),_num_parameters(self.params_p_logstd_s))
        return msg[:-1]

        
    def setup_parameters(self):        
        self.num_params_ps = _num_parameters(self.prestochastic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        self.params_q_logstd_ps =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.prestochastic_nn.parameters()])
        self.params_p_mean_ps = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.prestochastic_nn.parameters()] )
        self.params_p_logstd_ps = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.prestochastic_nn.parameters()]) 

        self.params_q_logstd_s =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
        self.params_p_mean_s = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd_s = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        return    
    
    
    def setup_functiondist(self):    
        func_model_ps, _ , buffers_ps = make_functional_with_buffers(self.prestochastic_nn)                
        func_model_s, _ ,  buffers_s = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        
        self.func_model_ps =  lambda params, x_in : func_model_ps(params, buffers_ps, x_in) 
        self.func_model_s =   lambda params, x_in : func_model_s(params, buffers_s, x_in) 
        
        self.flag_setup = True         
        return         


    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model_s,argnums=0)(q_params, x)                
            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)
    
        
    def sample_weight(self, param_q_mean, param_q_logstd):        
        weight_sample = []
        for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
            ith_weight_sample = ith_q_mean + transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)  
            weight_sample.append(ith_weight_sample)
        return weight_sample
        
                        
    def compute_func_samples(self, x, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s, nfsample=5):                
        self.setup_functiondist() if not self.flag_setup else 1
                
        func_samples = []
        for _ in range(nfsample):
            ith_weight_sample_ps = self.sample_weight(param_q_mean_ps, param_q_logstd_ps)            
            ith_h_samples = self.func_model_ps(ith_weight_sample_ps , x)                            
            
            ith_weight_sample_s = self.sample_weight(param_q_mean_s, param_q_logstd_s)                        
            ith_func_samples = self.func_model_s(ith_weight_sample_s , ith_h_samples)                                        
            func_samples.append(  ith_func_samples[None,...]  )  

        func_samples = torch.cat(func_samples,dim=0) #(nfsample,nb,ydim)           
        
        return func_samples



    #def compute_func_dist(self, x , param_q_mean, param_q_logstd ):
    def compute_func_dist(self, x, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s):                
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        
        weight_sample_ps = self.sample_weight(param_q_mean_ps, param_q_logstd_ps)            
        h_samples = self.func_model_ps(weight_sample_ps , x)                            
  
        #weight_sample_s = self.sample_weight(param_q_mean_s, param_q_logstd_s)                        
        #func_samples = self.func_model_s(weight_sample_s , h_samples)                                        
        f_mean = self.func_model_s(param_q_mean_s , h_samples)                                        
 
        # compute f_Lcov
        with torch.no_grad():
            jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean_s, h_samples) #(nb,ydim,nparams))                
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd_s) )
        param_q_std_vec    = transform_logstd(param_q_logstd_vec) #(nparams)        
        f_Lcov = jacobian_f_vec * param_q_std_vec[None,None,:] #(nb,ydim,nparams)
        f_cov = f_Lcov @ f_Lcov.transpose(-1,-2) #(nb,ydim,ydim)
        return f_mean, f_cov
                
                            
                
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            #nfsample = configs.train.nfsample
            nfsample = 1

        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()                    
        # get variational params    
        param_q_mean_ps, param_q_logstd_ps  =  list(self.prestochastic_nn.parameters()) ,  list(self.params_q_logstd_ps)     
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)

        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
                                
            # get prior params    
            param_p_mean_ps, param_p_logstd_ps  = list(self.params_p_mean_ps) ,  list(self.params_p_logstd_ps)     
            param_p_mean_s, param_p_logstd_s    = list(self.params_p_mean_s) ,  list(self.params_p_logstd_s)       

            # compute functional variational distirbution on training set and context set
            qf_mean_x, qf_cov_x  = self.compute_func_dist(x, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s) #(nfsample,nb,nout)
            qf_mean_xc,qf_cov_xc = self.compute_func_dist(xc, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s) #(nfsample,nb,nout)

            # compute functional prior distirbution on training set and context set
            pf_mean_x, pf_cov_x  = self.compute_func_dist(x,  param_p_mean_ps, param_p_logstd_ps, param_p_mean_s, param_p_logstd_s) #(nfsample,nb,nout)
            pf_mean_xc,pf_cov_xc = self.compute_func_dist(xc, param_p_mean_ps, param_p_logstd_ps, param_p_mean_s, param_p_logstd_s) #(nfsample,nb,nout)

            pf_mean    , pf_cov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_cov_x,pf_cov_xc],dim=0)            
            qf_mean    , qf_cov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_cov_x,qf_cov_xc],dim=0)

            # compute kl
            sup_kl = self.compute_sup_kl( [qf_mean,qf_cov], [pf_mean,pf_cov], Lowrank=False, nctset=nctset) 
        else :
            sup_kl = torch.tensor(0.0).to(x.device)            
        #reg_kl = reglambda*sup_kl
        
            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = sup_kl                
   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1, use_bayes=False):        
        self.eval()        
        # get variational params    
        param_q_mean_ps, param_q_logstd_ps  =  list(self.prestochastic_nn.parameters()) ,  list(self.params_q_logstd_ps)     
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x, param_q_mean_ps, param_q_logstd_ps, param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        return outs       



    def compute_sup_kl(self, params_qf, params_pf, Lowrank=True, nctset=2, eps=1e-8):
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
                               
            i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank )        
            kl_divergence_list.append(i_kl)

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        return torch.clamp(kl_divergence_list.max(),min=eps)    
    
    
    
    
## ---------------------------------------------------------------------------------------------------------
## FBNN_RP_BASE
## ---------------------------------------------------------------------------------------------------------
class FBNN_RP_Base(FBNN_Base):
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):
        
        #super(FunctionalBNN_RP_Base,self).__init__(in_dim, out_dim, hidden_dim)
        super(FBNN_RP_Base,self).__init__()
    
        self.deterministic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo
                        

    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         
        
    def compute_log_density(self,params_q, logits_sample, asprior=False, Lowrank=True):
        log_qf = build_MVN(params_q, asprior=asprior, Lowrank=Lowrank ).log_prob(logits_sample) #(nfsample,nb)                        
        return log_qf
        
               
    def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        h = self.deterministic_nn(x)        
        f_mean = self.func_model(param_q_mean,h)  
        
        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
        sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
        std_on_fs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)        
        
        f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)        
        return f_mean, f_Lcov , std_on_fs
                   
                            
    def compute_func_samples(self, f_mean,f_Lcov, nfsample=5):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
        f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)    

        # compute f_samples
        #return f_mean, f_std_samples    
        f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)
        return f_samples.permute(2,0,1) #(nfsample,nb,nout)
        
        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1        
        
        #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
        self.train()            
        # get params
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        #print('pf x')
        #print('param_p_logstd {}'.format(param_p_logstd))                                
        pf_mean_x  ,pf_Lcov_x, pf_stdfs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd )     
        #print('qf x')        
        #print('param_q_logstd {}'.format(param_q_logstd))               
        qf_mean_x,  qf_Lcov_x, qf_stdfs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        logits_sample         = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
        
        ## compute log density
        #log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
        #log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            

        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
            # set functional dist on x u xc     
            #print('pf xc')            
            #print('param_p_logstd {}'.format(param_p_logstd))                                                                       
            pf_mean_xc ,pf_Lcov_xc, pf_stdfs_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )       
            #print('qf xc')            
            #print('param_q_logstd {}'.format(param_q_logstd))                                                     
            qf_mean_xc ,qf_Lcov_xc, qf_stdfs_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

            pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
            qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
            # compute kl
            sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
        else :
            sup_kl = torch.tensor(0.0).to(x.device)            
                    
       
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.log_qf,outs.log_pf = log_qf, log_pf
        outs.log_qf,outs.log_pf = None,None
        #outs.sup_kl = sup_kl                
        outs.reg = sup_kl           
        outs.stdfs = (pf_stdfs_x , qf_stdfs_x, pf_stdfs_xc , qf_stdfs_xc)       
                     
        return outs        
    
    
    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        qf_mean_x,qf_Lcov_x,qf_stdfs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        logits_sample = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        outs.qfLcov = qf_Lcov_x  
        outs.stdfs = qf_stdfs_x       
        return outs         
                

    def compute_sup_kl(self, params_qf,params_pf, Lowrank=True, nctset=2, eps=1e-4):
        # set params
        qf_mean,qf_Lcov = params_qf
        pf_mean,pf_Lcov = params_pf
        #zitter_mat = eps*torch.ones_like(qf_mean).to(qf_mean.device)        

        # set ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_Lcov[i_set]
            i_params_p = pf_mean[i_set] , pf_Lcov[i_set]       
            i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank )        
            kl_divergence_list.append(i_kl)

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        return torch.clamp(kl_divergence_list.max(),min=eps_clamp)    



    # ## revised version        
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     # compute f_mean
    #     h = self.deterministic_nn(x)        
    #     f_mean = self.func_model(param_q_mean,h)  
        
    #     # compute f_Lcov
    #     jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)            
    #         #U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvdj') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
    #         S = torch.clamp(S, min=eps_eigv)
    #         U = torch.nan_to_num(U, nan=nan_val, posinf=posinf_val, neginf=neginf_val)
    #         S = torch.nan_to_num(S, nan=nan_val, posinf=posinf_val, neginf=neginf_val)
    #         VT = torch.nan_to_num(VT, nan=nan_val, posinf=posinf_val, neginf=neginf_val)
        
        
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
    #     sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
    #     eigen2_sum_VT2_qstd2 = torch.clamp( torch.pow(S,2)*sum_VT2_qstd2 ,min=nanstd_val )     #(nb,nydim)
    #     std_on_fs = torch.sqrt( eps_nan + eigen2_sum_VT2_qstd2 )
    #     # std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2 ).nan_to_num_(eps_nan) #(nb,nydim)        

    #     f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)        
    #     return f_mean, f_Lcov
        
    # ## revised version
    # def compute_func_samples(self, f_mean,f_Lcov, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
    #     eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #     f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)    

    #     # compute f_samples
    #     #return f_mean, f_std_samples    
    #     f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)
    #     f_samples = f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #     return torch.clamp(f_samples,min=eps_clamp,max=1.-eps_clamp) #(nfsample,nb,nout)
    
      
    # ## revised version
    # def predict_train(self, x , xc ,p_params=None,configs=None):                
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1
        
        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
                
    #     # compute logit samples
    #     #print('pf x')
    #     #print('param_p_logstd {}'.format(param_p_logstd))                        
    #     pf_mean_x  ,pf_Lcov_x =  self.compute_func_dist( x, param_p_mean, param_p_logstd )        
    #     #print('qf x')        
    #     #print('param_q_logstd {}'.format(param_q_logstd))                        
    #     qf_mean_x,  qf_Lcov_x =  self.compute_func_dist( x, param_q_mean, param_q_logstd )            
    #     pred_sample = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)        
    #     logits_sample = torch.log(pred_sample + eps_log)

    #     ## compute log density
    #     #log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
    #     #log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            
        
    #     # compute functional dist on context set        
    #     reglambda = float(reglambda)
    #     if reglambda > 0.0:            
    #         # set functional dist on x u xc                
    #         #print('pf xc')            
    #         #print('param_p_logstd {}'.format(param_p_logstd))                                    
    #         pf_mean_xc ,pf_Lcov_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
            
    #         #print('qf xc')            
    #         #print('param_q_logstd {}'.format(param_q_logstd))                                    
    #         qf_mean_xc ,qf_Lcov_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

    #         pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
    #         qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
    #         # compute kl
    #         sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
    #     else :
    #         sup_kl = torch.tensor(0.0).to(x.device)            
                    
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits = logits_sample
    #     # outs.log_qf = log_qf
    #     # outs.log_pf = log_pf       
    #     outs.log_qf = None
    #     outs.log_pf = None       

    #     #outs.sup_kl = sup_kl                
    #     outs.reg = sup_kl                        
    #     return outs       
        
    
    # def predict_eval(self, x , nfsample=1):        
    #     #print('nfsample: {} in eval'.format(nfsample))        
    #     self.eval()        
    #     # set params
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     qf_mean_x,qf_Lcov_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
    #     pred_samples = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
                                
    #     outs = AttrDict()
    #     outs.pred = pred_samples.mean(dim=0)  #(nb,nout)               
    #     outs.qfLcov = qf_Lcov_x         
    #     return outs       
    




## ---------------------------------------------------------------------------------------------------------
## FBNN_RP2_BASE : relaxation on RP
## ---------------------------------------------------------------------------------------------------------
class FBNN_RP2_Base(FBNN_RP_Base):
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):
        
        #super(FunctionalBNN_RP_Base,self).__init__(in_dim, out_dim, hidden_dim)
        super(FBNN_RP2_Base,self).__init__()
    
        self.deterministic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo
                        

    def sample_weight(self, param_q_mean, param_q_logstd):        
        weight_sample,delta_sample = [],[]
        for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
            ith_q_sample = transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)
            ith_weight_sample = ith_q_mean +  ith_q_sample                      
            delta_sample.append( -ith_q_sample)
            weight_sample.append(ith_weight_sample)
        return weight_sample, delta_sample

       
    def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        h = self.deterministic_nn(x)                
        if self.training:
            param_q_sample, delta_sample = self.sample_weight(param_q_mean, param_q_logstd)        
            f_mean0 = self.func_model(param_q_sample, h)          
            
            jacobian_f_vec = self.compute_jacobian_f_vec(param_q_sample, h)   #(nb,nout,nparams)        
            delta_sample_vec = parameters_to_vector(delta_sample) #(nparams)
            f_delta = (jacobian_f_vec*delta_sample_vec[None,None,:]).sum(dim=-1) #(nb,nout)
                    
            f_mean = f_mean0 +  f_delta
            #print('delta_sample_vec.shape: {}, f_delta.shape: {}, f_mean.shape: {}'.format(delta_sample_vec.shape,f_delta.shape,f_mean.shape))
        else:
            #print('eval mode')
            f_mean = self.func_model(param_q_mean, h)          
            jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)   #(nb,nout,nparams)        
                                
        # compute f_Lcov        
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                         
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
        sum_VT2_qstd2 = ( (VT * param_q_std_vec)**2 ).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = torch.clamp( (S**2)*sum_VT2_qstd2 , min=eps_clamp ).nan_to_num_(eps_nan)  #(nb,nydim)
        std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2 ) #(nb,nydim)        
        f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)
        
        return f_mean, f_Lcov, std_on_fs 
                
        
    # def compute_func_samples(self, f_mean,f_Lcov, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
    #     eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #     f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)    

    #     # compute f_samples
    #     f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)
    #     return f_samples.permute(2,0,1) #(nfsample,nb,nout)
                
        
        
        
        
        
        
        
        


## ---------------------------------------------------------------------------------------------------------
## FBNN_RP_Branch_BASE
## ---------------------------------------------------------------------------------------------------------

class FBNN_RP_Branch_Base(FBNN_RP_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(FBNN_RP_Branch_Base, self).__init__(scalelogstd)
        
        self.deterministic_base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None   
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo
        

    def __repr__(self):        
        msg = 'deterministic base NN: {} \n'.format(self.deterministic_base_nn)
        msg += 'deterministic NN: {} \n'.format(self.deterministic_nn)        
        msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]

                           

    def compute_func_dist(self, x , param_q_mean, param_q_logstd, eps=1e-8 ):
    #def compute_lastfeature_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        h = self.deterministic_base_nn(x)                
        f_mean_base = self.deterministic_nn(h)
        f_mean = self.func_model(param_q_mean,h)  

        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h.detach().clone())        
        # U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)        
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                         
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
        sum_VT2_qstd2 = ( (VT * param_q_std_vec)**2 ).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = (S**2)*sum_VT2_qstd2 #(nb,nydim)
        std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2.clamp_(min=eps)) #(nb,nydim)        
        f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)
        #return f_mean, f_Lcov
        return f_mean, f_Lcov, f_mean_base
        

    
    #def predict_train(self, x , xc , nfsample=10 , p_params=None, nctset= 1, reglambda=0.1):
    def predict_train(self, x , xc ,  p_params=None, configs = None):        
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()            
        # get params
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        pf_mean_x  ,pf_Lcov_x,           _  =  self.compute_func_dist( x, param_p_mean, param_p_logstd )        
        qf_mean_x,  qf_Lcov_x, qf_mean_d_x  =  self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        logits_sample = self.compute_func_samples(qf_mean_d_x, qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)

        #compute feature mathtcing regloss
        reg_matching = reglambda*(qf_mean_x - qf_mean_d_x.detach().clone() ).norm(dim=-1).mean()
        #reg_matching = 1*(qf_mean_x - qf_mean_d_x.detach().clone() ).norm(dim=-1).mean()


        # compute log density
        log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
        log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            

        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
            # set functional dist on x u xc                            
            pf_mean_xc ,pf_Lcov_xc, _ = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
            qf_mean_xc ,qf_Lcov_xc, _ = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

            pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
            qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
            # compute kl
            sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
        else :
            sup_kl = torch.tensor(0.0).to(x.device)            
        reg_kl = reglambda*sup_kl
            
        
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        outs.log_qf = log_qf
        outs.log_pf = log_pf       
        outs.sup_kl = sup_kl
        #outs.reg_matching = reg_matching                
        outs.reg = (reg_kl,reg_matching)                        
        return outs        

        
        
    def predict_eval(self, x , nfsample=1):        
            
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        qf_mean_x,  qf_Lcov_x, qf_mean_d_x  = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        #logits_sample = qf_mean_d_x[None,...]   #(1,nb,nout)     
        logits_sample = self.compute_func_samples(qf_mean_d_x, qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
                                               
        pred = F.softmax(logits_sample,dim=-1) #(1+nfsample,nb,nout) 
                                
        outs = AttrDict()        
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        return outs                   
    

    

# ## ---------------------------------------------------------------------------------------------------------
# ## FBNN_RP_Branch_BASE
# ## ---------------------------------------------------------------------------------------------------------
# class FBNN_RP_Branch_Base(FBNN_RP_Base):
#     #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
#     def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
#         super(FBNN_RP_Branch_Base, self).__init__(scalelogstd)
        
#         self.deterministic_nn_base = None
#         self.deterministic_nn = None
#         self.stochastic_nn = None   
#         self.scalelogstd = scalelogstd        
#         self.flag_setup = False
#         self.losstype = losstype # elbo, iwae, sumo
        

#     def __repr__(self):        
#         msg = 'deterministic base NN: {} \n'.format(self.deterministic_base_nn)
#         msg += 'deterministic NN: {} \n'.format(self.deterministic_nn)        
#         msg += 'stochastic NN: {} \n'.format(self.stochastic_nn)       
#         msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
#         msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
#         msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
#         return msg[:-1]

                           

#     def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
#     #def compute_lastfeature_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
        
#         self.setup_functiondist() if not self.flag_setup else 1
        
#         # compute f_mean
#         h = self.deterministic_base_nn(x)                
#         f_mean_base = self.deterministic_nn(h)
#         f_mean = self.func_model(param_q_mean,h)  

#         # compute f_Lcov
#         jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h.detach().clone())        
#         # U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)        
#         with torch.no_grad():        
#             U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                         
#         param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
#         param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
#         sum_VT2_qstd2 = ( (VT * param_q_std_vec)**2 ).sum(dim=-1)  #(nb,nydim)
#         eigen2_sum_VT2_qstd2 = (S**2)*sum_VT2_qstd2 #(nb,nydim)
#         std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2.clamp_(min=eps)) #(nb,nydim)        
#         f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)
#         #return f_mean, f_Lcov
#         return f_mean, f_Lcov, f_mean_base
        
        
#     # def compute_func_samples(self, f_mean,f_Lcov, nfsample=5):        
#     #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
#     #     nb,ydim = f_mean.shape
#     #     eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
#     #     f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)    

#     #     # compute f_samples
#     #     f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)
#     #     return f_samples.permute(2,0,1) #(nfsample,nb,nout)
        
        
            
#     #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):
#     def predict_train(self, x , xc , nfsample=10 , p_params=None, nctset= 1, reglambda=0.1):
        
#         self.train()            
#         # get params
#         param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
#         param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
#         # compute logit samples
#         pf_mean_x  ,pf_Lcov_x, _  =  self.compute_func_dist( x, param_p_mean, param_p_logstd )        
#         qf_mean_x,  qf_Lcov_x, qf_mean_d_x  = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
#         logits_sample_d = qf_mean_d_x.unsqueeze(0).repeat(nfsample,1,1)
#         logits_sample_s = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
#         logits_sample = torch.cat([logits_sample_d, logits_sample_s],dim=0) #(2*nfsample,nb,nout)

#         #compute feature mathtcing regloss
#         reg_matching = (qf_mean_x - qf_mean_d_x).norm(dim=-1).mean()


#         # compute log density
#         log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
#         log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            

#         # compute functional dist on context set        
#         reglambda = float(reglambda)
#         if reglambda > 0.0:            
#             # set functional dist on x u xc                            
#             pf_mean_xc ,pf_Lcov_xc, _ = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
#             qf_mean_xc ,qf_Lcov_xc, _ = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

#             pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
#             qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
#             # compute kl
#             sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
#         else :
#             sup_kl = torch.tensor(0.0).to(x.device)            
#         reg_kl = reglambda*sup_kl
            
        
#         # outputs        
#         outs = AttrDict()
#         outs.logits = logits_sample
#         outs.log_qf = log_qf
#         outs.log_pf = log_pf       
#         outs.sup_kl = sup_kl
#         #outs.reg_matching = reg_matching                
#         outs.reg = (reg_kl,reg_matching)                        
#         return outs        

        
#     def predict_eval(self, x , nfsample=1):        
#         self.eval()        
#         # set params
#         param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
#         # compute logit samples
#         qf_mean_x,  qf_Lcov_x, qf_mean_d_x  = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
#         #logits_sample_d = qf_mean_d_x.unsqueeze(0).repeat(nfsample,1,1)
#         logits_sample_d = qf_mean_d_x[None,...]   #(1,nb,nout)     
#         logits_sample_s = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
#         logits_sample = torch.cat([logits_sample_d, logits_sample_s],dim=0) #(1+nfsample,nb,nout)
                                               
#         pred = F.softmax(logits_sample,dim=-1) #(1+nfsample,nb,nout) 
                                
#         outs = AttrDict()        
#         outs.pred = pred.mean(dim=0)  #(nb,nout)               
#         outs.logits = logits_sample
#         return outs               

## ---------------------------------------------------------------------------------------------------------
## FDROPOUT_BASE: Dropout on function space regularization
## ---------------------------------------------------------------------------------------------------------    

class FDROPOUT_Base(nn.Module):
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
        
        super(FDROPOUT_Base,self).__init__()
    
        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

    #def extra_repr(self):
    def __repr__(self):        
        msg = 'scalelogstd: {} \n'.format(self.scalelogstd) 
        #msg += '#qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd))
        #msg += '#pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean),_num_parameters(self.params_p_logstd))
        return msg[:-1]


    def setup_parameters(self):        
        raise NotImplementedError
    
    def _get_noiseparams(self,option='q'):
        raise NotImplementedError

    def _get_noiseparam(self, option='q'):
        if option == 'q':
            param_mean   = [self.param_q_mean1    , self.param_q_mean2]
            param_logstd = [self.param_q_logstd1  , self.param_q_logstd2]
            
        if option == 'p':
            param_mean   = [self.param_p_mean1    , self.param_p_mean2]
            param_logstd = [self.param_p_logstd1  , self.param_p_logstd2]
         
        return param_mean, param_logstd

    def compute_jacobian_f_vec(self, x, param_mean ):
        raise NotImplementedError
    
    def compute_func_mean(self, x, param_mean):
        raise NotImplementedError


    def compute_func_dist(self, x , param_q_mean, param_q_logstd ):
        #compute f_mean    
        f_mean = self.compute_func_mean(x, param_q_mean)

        # compute f_Lcov
        with torch.no_grad():
            jacobian_f_vec = self.compute_jacobian_f_vec(x, param_q_mean  )                
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec    = transform_logstd(param_q_logstd_vec) #(nparams)
        #print('jacobian_f_vec.shape: {}'.format(jacobian_f_vec.shape))        
        f_Lcov = jacobian_f_vec * param_q_std_vec[None,None,:] #(nb,ydim,nparams)
        f_cov = f_Lcov @ f_Lcov.transpose(-1,-2) 
        
        #breakpoint()
        #f_cov = f_cov + (zitter_mat*torch.eye(f_cov.shape[-1])[None,...]).to(x.device)  #(nb,ydim,ydim)
        return f_mean, f_cov


    
    # #def compute_func_dist(self, x , param_q_mean, param_q_logstd ):
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd, zitter_mat=1e-4 ):
    #     #compute f_mean    
    #     f_mean = self.compute_func_mean(x, param_q_mean)

    #     # compute f_Lcov
    #     with torch.no_grad():
    #         jacobian_f_vec = self.compute_jacobian_f_vec(x, param_q_mean  )                
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec    = transform_logstd(param_q_logstd_vec) #(nparams)
        
    #     #print('jacobian_f_vec.shape: {}'.format(jacobian_f_vec.shape))        
    #     f_Lcov = jacobian_f_vec * param_q_std_vec[None,None,:] #(nb,ydim,nparams)
    #     f_cov = f_Lcov @ f_Lcov.transpose(-1,-2) 
    #     f_cov = f_cov + (zitter_mat*torch.eye(f_cov.shape[-1])[None,...]).to(x.device)  #(nb,ydim,ydim)
    #     return f_mean, f_cov
    
                  
                

    def compute_sup_kl(self, params_qf, params_pf, Lowrank=False, nctset=2, eps=1e-8):
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
            i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank )        
            kl_divergence_list.append(i_kl)

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        #breakpoint()
        return torch.clamp(kl_divergence_list.max(),min=eps_clamp)    
                            
                
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()                    
        # get params
        # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
        # param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = self._get_noiseparam('p')  
        param_q_mean, param_q_logstd  = self._get_noiseparam('q')     
                
        # compute function samples
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)

        # compute functional dist on context set        
        reglambda = float(reglambda)
        if reglambda > 0.0:            
            # set functional dist on x u xc                            
            pf_mean_x ,  pf_cov_x = self.compute_func_dist( x, param_p_mean, param_p_logstd )            
            qf_mean_x ,  qf_cov_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
            
            pf_mean_xc , pf_cov_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
            qf_mean_xc , qf_cov_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

            pf_mean    , pf_cov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_cov_x,pf_cov_xc],dim=0)            
            qf_mean    , qf_cov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_cov_x,qf_cov_xc],dim=0)
        
            #breakpoint()

            # compute kl
            sup_kl = self.compute_sup_kl([qf_mean, qf_cov], [pf_mean, pf_cov], Lowrank=False, nctset=nctset) 
        else :
            sup_kl = torch.tensor(0.0).to(x.device)            

            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = sup_kl                
   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1):        
        self.eval()        
        # get params
        #param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()),  list(self.params_q_logstd)     
        param_q_mean, param_q_logstd  = self._get_noiseparam('q')     
        
        # compute function samples
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        return outs       
        
    
    # #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    # def predict_train(self, x , xc ,p_params=None,configs=None):                
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1
        
        
    #     self.train()                    
    #     # get params
    #     # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
    #     # param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
    #     param_p_mean, param_p_logstd  = self._get_noiseparam('p')  
    #     param_q_mean, param_q_logstd  = self._get_noiseparam('q')     
                
    #     # compute function samples
    #     #logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)
    #     pred_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)
    #     logits_sample = torch.log(pred_sample+eps_log)

    #     # compute functional dist on context set        
    #     reglambda = float(reglambda)
    #     if reglambda > 0.0:            
    #         # set functional dist on x u xc                            
    #         pf_mean_x ,  pf_cov_x = self.compute_func_dist( x, param_p_mean, param_p_logstd )            
    #         qf_mean_x ,  qf_cov_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
            
    #         pf_mean_xc , pf_cov_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )            
    #         qf_mean_xc , qf_cov_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

    #         pf_mean    , pf_cov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_cov_x,pf_cov_xc],dim=0)            
    #         qf_mean    , qf_cov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_cov_x,qf_cov_xc],dim=0)
        
    #         breakpoint()
    #         # compute kl
    #         sup_kl = self.compute_sup_kl([qf_mean, qf_cov], [pf_mean, pf_cov], Lowrank=False, nctset=nctset) 
    #     else :
    #         sup_kl = torch.tensor(0.0).to(x.device)            
    #     #reg_kl = reglambda*sup_kl
        
            
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits = logits_sample
    #     #outs.sup_kl = sup_kl                
    #     outs.reg = sup_kl                
   
    #     return outs        
       
            
    # def predict_eval(self, x , nfsample=1):        
    #     self.eval()        
    #     # get params
    #     #param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()),  list(self.params_q_logstd)     
    #     param_q_mean, param_q_logstd  = self._get_noiseparam('q')     
        
    #     # compute function samples
    #     pred = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)                
    #     #pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)               
    #     return outs           
    
    
    
    
    





##---------------------------------------------------------------------------------------------------------
## FBNN_RPAwareclscat_Base
##---------------------------------------------------------------------------------------------------------

class FBNN_RPAwareclscat_Base(FBNN_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10, losstype='elbo'):
    def __init__(self, priorstd_in=1e-1, priorstd_out=5e-1, scalelogstd = -8, losstype='elbo'):
                        
        super(FBNN_RPAwareclscat_Base, self).__init__( scalelogstd, losstype)

        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

        self.priorstd_in  =  priorstd_in               
        self.priorstd_out =  priorstd_out       
        
                
        self.base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None
        self.static_mh = None
        # self.static_mh = Static_CatLayersfeature_MH_classifer( nfeature_list=[16,32,64],  
        #                                                         nlabel=out_dim 
        #                                                     )


        self.usemapestimator = False
        self.setprior=False
        
        # setup variance parameters
        #self.setup_parameters()
        
        
        
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         
        
        
    def setup_parameters(self):        
        #self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        #self.params_logstd = nn.Parameter(torch.empty(self.num_params_s).uniform_(-5,-4))    
        #self.params_q_logstd = nn.Parameter(torch.empty(self.num_params_s).uniform_(self.scalelogstd,self.scalelogstd+1) )    
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        #self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )        
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,0+1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(-1,0)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

        return   
            

    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                
            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)
    

    def _compute_stdonfs(self, param_q_logstd, U, S, VT):
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
        sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
        stdonfs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)                
        return stdonfs

    
    def base_nn_aux(self, x):
        return self.base_nn(x)

    def deterministic_nn_aux(self, h):
        h_feature_list = []
        for ith_module in self.deterministic_nn:
            h = ith_module(h)
            h_feature_list.append(h.view(h.size(0),h.size(1),-1 ).mean(dim=-1) )
        return h, h_feature_list
    

    #def compute_func_dist(self, x , param_q_mean, param_q_logstd, current_epoch=1, update_mh_static=False, datatype='x', intrain=True):
    def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x' , append=True   ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        ## compute f_mean
        h = self.base_nn(x)
        h, h_list = self.deterministic_nn_aux(h)        
        f_mean = self.func_model(param_q_mean,h)  
        

        if not self.usemapestimator:            
                        
            # compute f_Lcov
            jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
            with torch.no_grad():        
                U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)

            # compute std on fs 
            stdonfs = self._compute_stdonfs(param_q_logstd, U, S, VT)
            
            
            
            ## -----------------------------------------------------------------------------------------------------------
            ## compute indist score to make predicive distribtuion
            
            #if self.static_mh.usemhbias:                
            indist_score = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
            #breakpoint()
            # reflect in-distribution prior on function space inference
            #stdonfs = stdonfs*indist_score        #v11  --> failed, learnable together, --> inscore x == inscore xc 라는 결과가나옴,      
            #stdonfs = stdonfs*(1 + indist_score)   #v12  --> failed, learnable together, --> inscore x == 이또한 inscore xc 라는 결과가나옴,          
            #stdonfs = stdonfs*(1 + indist_score.detach().clone())   #v13  -->
            ## -----------------------------------------------------------------------------------------------------------
            #stdonfs = stdonfs*(1 + indist_score.detach().clone())   #v13  -->

            # compute f_Lcov 
            f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     
            
            
            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = f_Lcov
            outs.stdonfs = stdonfs
            outs.stdonfs_z = stdonfs*(1 + indist_score.detach().clone())            
            outs.h_list = h_list
            outs.indist_score = indist_score

        else:
            indist_score = None
            
            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = None
            outs.stdonfs = None
            outs.stdonfs_z = None
            
            outs.h_list = h_list
            outs.indist_score = indist_score
                       
        return outs    
    
                            
    def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        

        # compute f_samples
        if not self.usemapestimator:           
            
            z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
            z_samples = z_samples[:,None,:]                     
            eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)

            f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
            f_std_samples = f_std_samples*(1 + z_samples)    

            f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)                        
            return f_samples.permute(2,0,1) #(nfsample,nb,nout)
        else:        
            return f_mean[None,...] #(1,nb,nout)
        

    #def predict_train(self, x , xc ,p_params=None,configs=None):                
    def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1        
                
        #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
        self.train()            
        # get params
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute functional dist on train set
        q_outs_x         = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x', append=True)     
        #logits_sample    = self.compute_func_samples(q_outs_x.f_mean, q_outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
        logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        
                
        # compute functional dist on context set        
        if not self.usemapestimator:            
            reglambda = float(reglambda)
            
            # compute functional dist on context set
            q_outs_xc     = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc', append=True)                       
            # compute stdonfs
            stdfs = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
            # compute indist score        
            indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score) 


            qf_mean_x   = q_outs_x.f_mean
            #qf_std_x    = q_outs_x.stdonfs
            qf_std_x    = q_outs_x.stdonfs_z

            qf_mean_xc  = q_outs_xc.f_mean
            #qf_std_xc   = q_outs_xc.stdonfs
            qf_std_xc   = q_outs_xc.stdonfs_z
            
            ## set variational dist on xc
            qf_mean   ,qf_Diag   = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_std_x,qf_std_xc],dim=0)            
            
                    
            
            ## set prior on x 
            p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x', append=False)            
            pf_mean_x_in  = p_outs_x.f_mean            
            pf_std_x_in   = self.priorstd_in*torch.ones_like(pf_mean_x_in).to(xc.device)
                        
            ## set prior on xc 
            p_outs_xc       = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc', append=False)            
            pf_mean_xc_in   =  p_outs_xc.f_mean             
            pf_std_xc_in    =  self.priorstd_in*torch.ones_like(pf_mean_xc_in).to(xc.device)

            pf_mean_xc_out  =  torch.zeros_like(qf_mean_xc).to(xc.device)        
            pf_std_xc_out   =  self.priorstd_out*torch.ones_like(pf_mean_xc_out).to(xc.device)

            
            ## set prior on x and xc            
            pf_mean_x ,pf_std_x = pf_mean_x_in , pf_std_x_in            

            z_xc = torch.bernoulli(p_outs_xc.indist_score).to(xc.device).detach().clone()             
            pf_mean_xc =  (1 - z_xc)*pf_mean_xc_in  +  z_xc*pf_mean_xc_out 
            pf_std_xc  =  (1 - z_xc)*pf_std_xc_in   +  z_xc*pf_std_xc_out
 
            pf_mean, pf_Diag = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_std_x,pf_std_xc],dim=0) 
            
            #breakpoint()
            
            ## set indist score on x and xc
            #indist_score = torch.cat([ q_outs_x.indist_score, q_outs_xc.indist_score],dim=0)
            sup_kl = self.compute_sup_kl( (qf_mean, qf_Diag), (pf_mean, pf_Diag), Diag=True, nctset=nctset) 
            
            
            ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
            self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                    
            
            
        else :            
            # compute stdonfs
            #stdfs = (q_outs_x.stdonfs, None)  
            stdfs = None                         
            # compute indist score            
            indistscore = None
            #q_outs_xc = None
            #sup_kl = torch.tensor(0.0).to(x.device)      
            
            # l2 regularization on param
            param_vec = parameters_to_vector(self.deterministic_nn.parameters())
            sup_kl = 0.5 * torch.norm(param_vec, p = 2)

                           
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.log_qf,outs.log_pf = log_qf, log_pf
        outs.log_qf,outs.log_pf = None,None
        #outs.sup_kl = sup_kl                
        outs.reg = sup_kl           
        outs.stdfs = stdfs   
        outs.indistscore = indistscore 
        return outs                
                
                    
                

    def compute_sup_kl(self, params_qf,params_pf, Lowrank=True, Diag=True, nctset=2, eps=1e-4):
        
        def compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=True)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=True)
            #breakpoint()
            return kl_divergence(qf_dist,pf_dist).sum(dim=-1,keepdim=True).mean(dim=0)
    
        
        # set params
        qf_mean,qf_Lcov = params_qf
        pf_mean,pf_Lcov = params_pf
        #zitter_mat = eps*torch.ones_like(qf_mean).to(qf_mean.device)        

        # set ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_Lcov[i_set]
            i_params_p = pf_mean[i_set] , pf_Lcov[i_set]       

            i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=True )                    
            kl_divergence_list.append(i_kl)
 
        #breakpoint()
        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        return torch.clamp(kl_divergence_list.max(),min=eps_clamp)    



    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x', append=False)           
        #logits_sample = self.compute_func_samples(q_outs_x.f_mean,outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
        logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        outs.stdfs = q_outs_x.stdonfs
        return outs                 
                                                           
        
