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


# def compute_kl2(params_qf, params_pf , Lowrank=True, Diag=False):
#     qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
#     pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
#     #breakpoint()
#     return kl_divergence(qf_dist,pf_dist).sum(dim=-1,keepdim=True)


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



##---------------------------------------------------------------------------------------------------------
##  BNN_MFVI_BASE Revistied
##---------------------------------------------------------------------------------------------------------
class BNN_Base_Revisited(nn.Module):    
    #def __init__(self, scalelogstd = -8 ,losstype = 'elbo'):        
    def __init__(self, scalelogstd = -8):        
        
        super(BNN_Base_Revisited,self).__init__()
    
        #self.prestochastic_nn = None
        self.stochastic_nn = None   
        self.flag_setup = False
        
        self.scalelogstd = scalelogstd        
        
        #self.losstype = losstype # elbo, iwae, sumo

    #def extra_repr(self):
    def __repr__(self):        
        #msg = 'prestohcastic NN: {} \n'.format(self.prestochastic_nn)
        msg = 'stochastic NN: {} \n'.format(self.stochastic_nn)       
        msg += 'scalelogstd: {} \n'.format(self.scalelogstd) 
        #msg += '#pre stochastic qmu: {}, #qstd: {} \n'.format(_num_parameters(self.prestochastic_nn),_num_parameters(self.params_q_logstd_ps))        
        #msg += '#pre stochastic pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_ps),_num_parameters(self.params_p_logstd_ps))              
        msg += '#stochastic qmu: {}, #qstd: {} \n'.format(_num_parameters(self.stochastic_nn),_num_parameters(self.params_q_logstd_s))
        msg += '#stochastic pmu: {}, #pstd: {} \n'.format(_num_parameters(self.params_p_mean_s),_num_parameters(self.params_p_logstd_s))
        return msg[:-1]

        
    def setup_parameters(self):        
        #self.num_params_ps = _num_parameters(self.prestochastic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  

        #self.params_q_logstd_ps =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.prestochastic_nn.parameters()])
        #self.params_p_mean_ps = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.prestochastic_nn.parameters()] )
        #self.params_p_logstd_ps = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.prestochastic_nn.parameters()]) 

        self.params_q_logstd_s =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
        self.params_p_mean_s = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        #self.params_p_logstd_s = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        self.params_p_logstd_s = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) ##that is solution, will be a solution of MFVI as well.

        return    
    
    
    def setup_functiondist(self):    
        #func_model_ps, _ , buffers_ps = make_functional_with_buffers(self.prestochastic_nn)                
        func_model_s, _ ,  buffers_s = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        
        #self.func_model_ps =  lambda params, x_in : func_model_ps(params, buffers_ps, x_in) 
        self.func_model_s =   lambda params, x_in : func_model_s(params, buffers_s, x_in) 
        
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
            nfsample = 1
            
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
            #weight_kl = self.compute_weight_kl(params_qf, params_pf)
            weight_kl = self.compute_weight_kl(params_qf, params_pf) / self.num_params_s

        else :
            weight_kl = torch.tensor(0.0).to(x.device)            
        #reg_kl = reglambda*sup_kl
        
            
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        #outs.sup_kl = sup_kl                
        outs.reg = weight_kl               
   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1, topk=1):        
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
    
    
    def forward(self,x, nfsample = 1):
        # get variational params    
        param_q_mean_s, param_q_logstd_s    =  list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd_s)     

       # compute function samples        
        logits_sample  = self.compute_func_samples(x, param_q_mean_s, param_q_logstd_s , nfsample=nfsample) #(nfsample,nb,nout)                
        return logits_sample        
    



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

        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(0,1)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) ##that is solution, will be a solution of MFVI as well.
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
       
            
    def predict_eval(self, x , nfsample=1 , topk=1):        
        self.eval()        
        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()),  list(self.params_q_logstd)     
        
        # compute function samples
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)                
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample   #(nfsample,nb,ydim)      
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
    
    
    def foward(self,x,nfsample=1):
        # get params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute function samples
        logits_sample = self.compute_func_samples(x , param_q_mean, param_q_logstd, nfsample=nfsample) #(nfsample,nb,nout)
        return logits_sample
   
    
    
    
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
        outs.reg = sup_kl                   
        return outs        
       
            
    def predict_eval(self, x , nfsample=1 , topk=1):        
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
        
               
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     # compute f_mean
    #     h = self.deterministic_nn(x)        
    #     f_mean = self.func_model(param_q_mean,h)  
        
    #     # compute f_Lcov
    #     jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
    #     sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
    #     eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
    #     std_on_fs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)        
        
    #     f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)        
    #     return f_mean, f_Lcov , std_on_fs
                   
                            
    # def compute_func_samples(self, f_mean,f_Lcov, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
    #     eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #     f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)    

    #     # compute f_samples
    #     #return f_mean, f_std_samples    
    #     f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)
    #     return f_samples.permute(2,0,1) #(nfsample,nb,nout)

        
        
    # def predict_train(self, x , xc ,p_params=None,configs=None):                
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
        
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     #print('pf x')
    #     #print('param_p_logstd {}'.format(param_p_logstd))                                
    #     pf_mean_x  ,pf_Lcov_x, pf_stdfs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd )     
    #     #print('qf x')        
    #     #print('param_q_logstd {}'.format(param_q_logstd))               
    #     qf_mean_x,  qf_Lcov_x, qf_stdfs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
    #     logits_sample                     = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
        
    #     ## compute log density
    #     #log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
    #     #log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            

    #     # compute functional dist on context set        
    #     reglambda = float(reglambda)
    #     if reglambda > 0.0:            
    #         # set functional dist on x u xc     
    #         #print('pf xc')            
    #         #print('param_p_logstd {}'.format(param_p_logstd))                                                                       
    #         pf_mean_xc ,pf_Lcov_xc, pf_stdfs_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )       
    #         #print('qf xc')            
    #         #print('param_q_logstd {}'.format(param_q_logstd))                                                     
    #         qf_mean_xc ,qf_Lcov_xc, qf_stdfs_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

    #         pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
    #         qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
    #         # compute kl
    #         sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
    #     else :
    #         sup_kl = torch.tensor(0.0).to(x.device)            
                    
       
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits = logits_sample
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     outs.reg = sup_kl           
    #     outs.stdfs = (pf_stdfs_x , qf_stdfs_x, pf_stdfs_xc , qf_stdfs_xc)       
                     
    #     return outs        
    
    
    # def predict_eval(self, x , nfsample=1):        
    #     #print('nfsample: {} in eval'.format(nfsample))        
    #     self.eval()        
    #     # set params
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     qf_mean_x,qf_Lcov_x,qf_stdfs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
    #     logits_sample = self.compute_func_samples(qf_mean_x,qf_Lcov_x, nfsample=nfsample) #(nfsample,nb,nout)
    #     pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)               
    #     outs.logits = logits_sample
    #     outs.qfLcov = qf_Lcov_x  
    #     outs.stdfs = qf_stdfs_x       
    #     return outs         
    
    
    



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
        
        #f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)        
        return f_mean, U , std_on_fs
    
    def compute_func_samples(self, f_mean, f_u, f_std, nfsample=5):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
        eps_samples = eps_samples* f_std[:,:,None] #(nb,ydim,nfsample)        
        f_std_samples = torch.einsum('bij,bjk->bik',f_u,eps_samples)    

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
                
        qf_mean_x,  qf_u_x, qf_std_x      = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        logits_sample                     = self.compute_func_samples(qf_mean_x,  qf_u_x, qf_std_x, nfsample=nfsample) #(nfsample,nb,nout)
        
        ## compute log density
        #log_qf = self.compute_log_density( [qf_mean_x,qf_Lcov_x], logits_sample, asprior=False, Lowrank = True )
        #log_pf = self.compute_log_density( [pf_mean_x,pf_Lcov_x], logits_sample, asprior=True,  Lowrank = True )
            

        # # compute functional dist on context set        
        # reglambda = float(reglambda)
        # if reglambda > 0.0:            
        #     # set functional dist on x u xc     
        #     #print('pf xc')            
        #     #print('param_p_logstd {}'.format(param_p_logstd))                                                                       
        #     pf_mean_xc ,pf_Lcov_xc, pf_stdfs_xc = self.compute_func_dist( xc, param_p_mean, param_p_logstd )       
        #     #print('qf xc')            
        #     #print('param_q_logstd {}'.format(param_q_logstd))                                                     
        #     qf_mean_xc ,qf_Lcov_xc, qf_stdfs_xc = self.compute_func_dist( xc, param_q_mean, param_q_logstd )            

        #     pf_mean    ,pf_Lcov = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_Lcov_x,pf_Lcov_xc],dim=0)            
        #     qf_mean    ,qf_Lcov = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_Lcov_x,qf_Lcov_xc],dim=0)
        
        #     # compute kl
        #     sup_kl = self.compute_sup_kl([qf_mean,qf_Lcov], [pf_mean,pf_Lcov], Lowrank=True, nctset=nctset) 
        # else :
        #     sup_kl = torch.tensor(0.0).to(x.device)            
        #breakpoint()
       
        # outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        outs.reg    = torch.tensor(0.0).to(x.device)                       
        
        # #outs.log_qf,outs.log_pf = log_qf, log_pf
        # outs.log_qf,outs.log_pf = None,None
        # #outs.sup_kl = sup_kl                
        # outs.reg = sup_kl           
        # outs.stdfs = (pf_stdfs_x , qf_stdfs_x, pf_stdfs_xc , qf_stdfs_xc)       
                     
        return outs        
    
    
    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        # compute logit samples
        qf_mean_x,  qf_u_x, qf_std_x      = self.compute_func_dist( x, param_q_mean, param_q_logstd )            
        logits_sample                     = self.compute_func_samples(qf_mean_x,  qf_u_x, qf_std_x, nfsample=nfsample) #(nfsample,nb,nout)
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        # outs.qfLcov = qf_Lcov_x  
        # outs.stdfs = qf_stdfs_x       
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

       
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     # compute f_mean
    #     h = self.deterministic_nn(x)                
    #     if self.training:
    #         param_q_sample, delta_sample = self.sample_weight(param_q_mean, param_q_logstd)        
    #         f_mean0 = self.func_model(param_q_sample, h)          
            
    #         jacobian_f_vec = self.compute_jacobian_f_vec(param_q_sample, h)   #(nb,nout,nparams)        
    #         delta_sample_vec = parameters_to_vector(delta_sample) #(nparams)
    #         f_delta = (jacobian_f_vec*delta_sample_vec[None,None,:]).sum(dim=-1) #(nb,nout)
                    
    #         f_mean = f_mean0 +  f_delta
    #         #print('delta_sample_vec.shape: {}, f_delta.shape: {}, f_mean.shape: {}'.format(delta_sample_vec.shape,f_delta.shape,f_mean.shape))
    #     else:
    #         #print('eval mode')
    #         f_mean = self.func_model(param_q_mean, h)          
    #         jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)   #(nb,nout,nparams)        
                                
    #     # compute f_Lcov        
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                         
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
    #     sum_VT2_qstd2 = ( (VT * param_q_std_vec)**2 ).sum(dim=-1)  #(nb,nydim)
    #     eigen2_sum_VT2_qstd2 = torch.clamp( (S**2)*sum_VT2_qstd2 , min=eps_clamp ).nan_to_num_(eps_nan)  #(nb,nydim)
    #     std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2 ) #(nb,nydim)        
    #     f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)
        
    #     return f_mean, f_Lcov, std_on_fs 
                
        
        
    def compute_func_dist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
        self.setup_functiondist() if not self.flag_setup else 1
        
        # compute f_mean
        h = self.deterministic_nn(x)                
        param_q_sample, delta_sample = self.sample_weight(param_q_mean, param_q_logstd)        
        f_mean0 = self.func_model(param_q_sample, h)          
        
        jacobian_f_vec   = self.compute_jacobian_f_vec(param_q_sample, h)   #(nb,nout,nparams)        
        delta_sample_vec = parameters_to_vector(delta_sample) #(nparams)
        f_delta          = (jacobian_f_vec*delta_sample_vec[None,None,:]).sum(dim=-1) #(nb,nout)
                
        f_mean = f_mean0 +  f_delta
                                
        # compute f_Lcov        
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                         
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                 
        sum_VT2_qstd2 = ( (VT * param_q_std_vec)**2 ).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = torch.clamp( (S**2)*sum_VT2_qstd2 , min=eps_clamp )  #(nb,nydim)
        std_on_fs = torch.sqrt( eigen2_sum_VT2_qstd2 ) #(nb,nydim)        
        f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)
        
        #breakpoint()
        # return f_mean, f_Lcov, std_on_fs 
        return f_mean, U, std_on_fs 
                

        
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
        
        self.cnt = 0       
        self.period_stochastic = 1        
        #self.period_stochastic = 2
        #self.period_stochastic = 3

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
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

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
                       
                       
        #print('outs.indist_score in compute dist: {}'.format(outs.indist_score)) 
                       
        return outs    
    


    def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        
        if self.training:
            # compute f_samples
            if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :           
                
                z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
                z_samples = z_samples[:,None,:]   #(nb,1,nfsample)                  
                eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)

                f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
                f_std_samples = f_std_samples*(1 + z_samples)    

                f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)                        
                return f_samples.permute(2,0,1) #(nfsample,nb,nout)
            else:        
                return f_mean[None,...] #(1,nb,nout)

        else:
            # compute f_samples
            if not self.usemapestimator:           
                
                z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
                z_samples = z_samples[:,None,:]                     
                #print('z_samples:{}'.format(z_samples))
                #print('z_samples:{}'.format(z_samples.shape))
                
                eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)

                f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
                f_std_samples = f_std_samples*(1 + z_samples)    

                f_samples = f_mean[...,None] + f_std_samples #(nb,nout,nfsample)                        
                return f_samples.permute(2,0,1) #(nfsample,nb,nout)
            else:        
                return f_mean[None,...] #(1,nb,nout)
            
        
        

    # #def predict_train(self, x , xc ,p_params=None,configs=None):                
    # def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
                
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute functional dist on train set
    #     q_outs_x         = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x', append=True)     
    #     #logits_sample    = self.compute_func_samples(q_outs_x.f_mean, q_outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
    #     logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        
                
    #     # compute functional dist on context set        
    #     if not self.usemapestimator:            
    #         reglambda = float(reglambda)
            
    #         # compute functional dist on context set
    #         q_outs_xc     = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc', append=True)                       
    #         # compute stdonfs
    #         stdfs = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
    #         # compute indist score        
    #         indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score) 


    #         qf_mean_x   = q_outs_x.f_mean
    #         #qf_std_x    = q_outs_x.stdonfs
    #         qf_std_x    = q_outs_x.stdonfs_z

    #         qf_mean_xc  = q_outs_xc.f_mean
    #         #qf_std_xc   = q_outs_xc.stdonfs
    #         qf_std_xc   = q_outs_xc.stdonfs_z
            
    #         ## set variational dist on xc
    #         qf_mean   ,qf_Diag   = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_std_x,qf_std_xc],dim=0)            
            
                    
            
    #         ## set prior on x 
    #         p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x', append=False)            
    #         pf_mean_x_in  = p_outs_x.f_mean            
    #         pf_std_x_in   = self.priorstd_in*torch.ones_like(pf_mean_x_in).to(xc.device)
                        
    #         ## set prior on xc 
    #         p_outs_xc       = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc', append=False)            
    #         pf_mean_xc_in   =  p_outs_xc.f_mean             
    #         pf_std_xc_in    =  self.priorstd_in*torch.ones_like(pf_mean_xc_in).to(xc.device)

    #         pf_mean_xc_out  =  torch.zeros_like(qf_mean_xc).to(xc.device)        
    #         pf_std_xc_out   =  self.priorstd_out*torch.ones_like(pf_mean_xc_out).to(xc.device)

            
    #         ## set prior on x and xc            
    #         pf_mean_x ,pf_std_x = pf_mean_x_in , pf_std_x_in            

    #         z_xc = torch.bernoulli(p_outs_xc.indist_score).to(xc.device).detach().clone()             
    #         pf_mean_xc =  (1 - z_xc)*pf_mean_xc_in  +  z_xc*pf_mean_xc_out 
    #         pf_std_xc  =  (1 - z_xc)*pf_std_xc_in   +  z_xc*pf_std_xc_out
 
    #         pf_mean, pf_Diag = torch.cat([pf_mean_x,pf_mean_xc],dim=0), torch.cat([pf_std_x,pf_std_xc],dim=0) 
            
    #         #breakpoint()
            
    #         ## set indist score on x and xc
    #         #indist_score = torch.cat([ q_outs_x.indist_score, q_outs_xc.indist_score],dim=0)
    #         sup_kl = self.compute_sup_kl( (qf_mean, qf_Diag), (pf_mean, pf_Diag), Diag=True, nctset=nctset) 
            
            
    #         ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
    #         self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                    
            
            
    #     else :            
    #         # compute stdonfs
    #         #stdfs = (q_outs_x.stdonfs, None)  
    #         stdfs = None                         
    #         # compute indist score            
    #         indistscore = None
    #         #q_outs_xc = None

    #         #sup_kl = torch.tensor(0.0).to(x.device)      

    #         # l2 regularization on param
    #         param_vec = parameters_to_vector(self.deterministic_nn.parameters())
    #         sup_kl = 0.5 * torch.norm(param_vec, p = 2)

                           
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits = logits_sample
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     outs.reg = sup_kl           
    #     outs.stdfs = stdfs   
    #     outs.indistscore = indistscore 
    #     return outs                
                
                
                
                
                    
    ##for fast training
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
        
        #breakpoint()
        self.cnt = current_epoch               
        #if not self.usemapestimator:            
        if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     
            
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
            
            
            ### update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
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
        #outs.indist_score = indistscore 
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
        outs.indist_score = q_outs_x.indist_score
        
        return outs                 
                                                           
        
        
        
        













##---------------------------------------------------------------------------------------------------------
## FBNN_RPAwareclscat_Base
##---------------------------------------------------------------------------------------------------------

class FBNN_RPLatAware_Base(FBNN_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10, losstype='elbo'):
    def __init__(self, priorstd_in=1e-1, priorstd_out=5e-1, scalelogstd = -8, losstype='elbo', updateperiod=1):
                        
        super(FBNN_RPLatAware_Base, self).__init__( scalelogstd, losstype)

        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

        self.priorstd_in  =  priorstd_in               
        self.priorstd_out =  priorstd_out       
        
                
        self.base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None
        self.static_mh = None

        
        self.usemapestimator = False
        self.setprior=False
        
        self.cnt = 0       
        self.period_stochastic = updateperiod
        #self.period_stochastic = 1        
        #self.period_stochastic = 2
        #self.period_stochastic = 3

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
        #self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd+1,self.scalelogstd+2)  ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

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
                        
            ## compute indist score to make predicive distribtuion            
            outs_mh = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
            # compute f_Lcov 
            f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     

            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = f_Lcov
            outs.stdonfs = stdonfs
            outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
            outs.h_list = h_list
            outs.indist_score = outs_mh.indist_score
            outs.pseudo_label = outs_mh.pseudo_label
            outs.mh_distance = outs_mh.mh_distance

        else:
            
            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = None
            outs.stdonfs = None
            outs.stdonfs_z = None
            
            outs.h_list = h_list
            outs.indist_score = None
            outs.pseudo_label = None                       
            outs.mh_distance = None                       
        return outs    
    


    # def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
    #     if self.training:
    #         # compute f_samples
    #         if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :           
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             z_samples = z_samples[:,None,:]   #(nb,1,nfsample)                  
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
    #             f_mean_samples = (f_mean[:,:,None]) #(nb,nout,1)

    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             #f_std_samples = f_std_samples*(1 + z_samples)    

    #             f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                        
    #             return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)

    #     else:
    #         # compute f_samples
    #         if not self.usemapestimator:           
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             z_samples = z_samples[:,None,:]                     #(nb,1,nfsample)                  
                
    #             ##(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                
    #             ## consider randomness for $z$, but makes generlaization worse                               
    #             f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) 

    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             ## consider randomness for $z$, but makes generlaization worse               
    #             f_std_samples = f_std_samples*(1 + z_samples)    

    #             f_samples = f_mean_samples + f_std_samples          #(nb,nout,nfsample)        
    #             return f_samples.permute(2,0,1)                     #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)
            
        


    # def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
    #     if self.training:
    #         # compute f_samples
    #         if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :           
    #             #z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             #z_samples = z_samples[:,None,:]   #(nb,1,nfsample)                  
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
    #             f_mean_samples = (f_mean[:,:,None]) #(nb,nout,1)
    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             #f_std_samples = f_std_samples*(1 + z_samples)    

    #             f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                        
    #             return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)

    #     else:
    #         # compute f_samples
    #         if not self.usemapestimator:           
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             z_samples = z_samples[:,None,:]                     #(nb,1,nfsample)                  
                
    #             ##(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                
    #             ## consider randomness for $z$, but makes generlaization worse                               
    #             f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) 

    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             ## consider randomness for $z$, but makes generlaization worse               
    #             f_std_samples = f_std_samples*(1 + z_samples)    

    #             f_samples = f_mean_samples + f_std_samples          #(nb,nout,nfsample)        
    #             return f_samples.permute(2,0,1)                     #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)                                
                    
                    
                    


    def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        
        if self.training:
            # compute f_samples
            if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :           
                #z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
                #z_samples = z_samples[:,None,:]   #(nb,1,nfsample)                  
                #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
                f_mean_samples = (f_mean[:,:,None]) #(nb,nout,1)
                eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
                f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
                #f_std_samples = f_std_samples*(1 + z_samples)    

                f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                        
                return f_samples.permute(2,0,1) #(nfsample,nb,nout)
            else:        
                return f_mean[None,...] #(1,nb,nout)

        else:
            # compute f_samples
            if not self.usemapestimator:           
                z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
                z_samples = z_samples[:,None,:]                     #(nb,1,nfsample)                  
                
                ##(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                                
                #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples)  ## consider randomness for $z$, but makes generlaization worse                               
                f_mean_samples = (f_mean[:,:,None]) 
                
                eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)                
                #f_std_samples = f_std_samples*(1 + z_samples) ## consider randomness for $z$, but makes generlaization worse                   
                f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)                

                f_samples = f_mean_samples + f_std_samples          #(nb,nout,nfsample)        
                return f_samples.permute(2,0,1)                     #(nfsample,nb,nout)
            else:        
                return f_mean[None,...] #(1,nb,nout)                                
                    
                    
                                        
                    
                    
                    
    ##for fast training
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
        
        #breakpoint()
        self.cnt = current_epoch               
        #if not self.usemapestimator:            
        if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     
            
            reglambda = float(reglambda)
            
            # compute functional dist on context set
            #if self.static_mh.usemhbias:
            #    breakpoint()
            q_outs_xc     = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc', append=True)                       
            # compute stdonfs
            stdfs = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
            # compute indist score        
            indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score) 
            pseudolabel = (q_outs_x.pseudo_label,q_outs_xc.pseudo_label)


            ## set variational dist on x and xc            
            qf_mean_x   = q_outs_x.f_mean
            qf_std_x    = q_outs_x.stdonfs
            qf_mean_xc  = q_outs_xc.f_mean
            qf_std_xc   = q_outs_xc.stdonfs            
            qf_mean   ,qf_Diag   = torch.cat([qf_mean_x,qf_mean_xc],dim=0), torch.cat([qf_std_x,qf_std_xc],dim=0)            
            
                    
            ## set prior on x and xc
            p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x', append=False)            
            pf_mean_x_in  = p_outs_x.f_mean            
            pf_std_x_in   = self.priorstd_in*torch.ones_like(pf_mean_x_in).to(xc.device)

            p_outs_xc       = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc', append=False)            
            pf_mean_xc_in   =  p_outs_xc.f_mean             
            pf_std_xc_in    =  self.priorstd_in*torch.ones_like(pf_mean_xc_in).to(xc.device)
            pf_mean_xc_out  =  torch.zeros_like(qf_mean_xc).to(xc.device)        
            pf_std_xc_out   =  self.priorstd_out*torch.ones_like(pf_mean_xc_out).to(xc.device)

            
            ## set latent prior and variational on xc            
            q_z_xc = torch.bernoulli(q_outs_xc.indist_score).to(xc.device).detach().clone()             
            a_z_xc = q_outs_xc.indist_score             
            
            ## discrete version
            qf_mean_xc_lat = (1 - q_z_xc)*qf_mean_xc
            qf_std_xc_lat  = (1 + q_z_xc)*qf_std_xc 
            pf_mean_xc_lat = (1 - q_z_xc)*pf_mean_xc_in  +  q_z_xc*pf_mean_xc_out 
            pf_std_xc_lat  = (1 - q_z_xc)*pf_std_xc_in   +  q_z_xc*pf_std_xc_out

            ## contig. relaxation version
            # qf_mean_xc_lat = (1 - a_z_xc)*qf_mean_xc
            # qf_std_xc_lat  = (1 + a_z_xc)*qf_std_xc 
            # pf_mean_xc_lat = (1 - q_z_xc)*pf_mean_xc_in  +  q_z_xc*pf_mean_xc_out 
            # pf_std_xc_lat  = (1 - q_z_xc)*pf_std_xc_in   +  q_z_xc*pf_std_xc_out

            ## compute sup_kl 
            sup_kl = self.compute_sup_kl( (qf_mean_xc_lat, qf_std_xc_lat), (pf_mean_xc_lat, pf_std_xc_lat), Diag=True, nctset=nctset) 
                        
            
            ### update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
            self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
            
        else :            
            # compute stdonfs
            #stdfs = (q_outs_x.stdonfs, None)  
            stdfs = None                         
            # compute indist score            
            indistscore = None
            pseudolabel = (None,None)
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
        #outs.indist_score = indistscore 
        outs.indistscore = indistscore
        outs.pseudolabel = pseudolabel
        
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
        outs.indist_score = q_outs_x.indist_score
        outs.mh_distance = q_outs_x.mh_distance 
        
        return outs                 
                                                           
        








##---------------------------------------------------------------------------------------------------------
## FBNN_RPAwareclscat_Base
##---------------------------------------------------------------------------------------------------------

import torch
from torch.distributions.beta import Beta
from torch.distributions.continuous_bernoulli import ContinuousBernoulli as CBERN

class FBNN_RPContiLatAware_Base(FBNN_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10, losstype='elbo'):
    def __init__(self, priorstd_in=1e-1, priorstd_out=5e-1, scalelogstd = -8, losstype='elbo', updateperiod=1 ,betahyp=.1):
                        
        super(FBNN_RPContiLatAware_Base, self).__init__( scalelogstd, losstype)

        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

        self.priorstd_in  =  priorstd_in               
        self.priorstd_out =  priorstd_out       
        
                
        self.base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None
        self.static_mh = None

        
        self.usemapestimator = False
        self.setprior=False
        
        self.cnt = 0       
        self.period_stochastic = updateperiod
        #self.period_stochastic = 1        
        #self.period_stochastic = 2
        #self.period_stochastic = 3

        # setup variance parameters
        #self.setup_parameters()

        self.betahyp = betahyp
        self.beta_dist = Beta(torch.tensor([betahyp]),torch.tensor([betahyp]))
        
        
        
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn)                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         
        
        
    def setup_parameters(self):        
        #self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )        
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

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
    




    # #def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x' , append=True   ):
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     ## compute f_mean
    #     h = self.base_nn(x)
    #     h, h_list = self.deterministic_nn_aux(h)        
    #     f_mean = self.func_model(param_q_mean,h)  
        

    #     if not self.usemapestimator:            
                        
    #         # compute f_Lcov
    #         jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
    #         with torch.no_grad():        
    #             U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)

    #         # compute std on fs 
    #         stdonfs = self._compute_stdonfs(param_q_logstd, U, S, VT)
                        
    #         ## compute indist score to make predicive distribtuion            
    #         outs_mh = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
    #         # compute f_Lcov 
    #         f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     

    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = f_Lcov
    #         outs.stdonfs = stdonfs
    #         #outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
    #         outs.h_list = h_list
    #         outs.indist_score = outs_mh.indist_score
    #         outs.pseudo_label = outs_mh.pseudo_label
    #         outs.mh_distance = outs_mh.mh_distance

    #     else:
            
    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = None
    #         outs.stdonfs = None
    #         #outs.stdonfs_z = None
            
    #         outs.h_list = h_list
    #         outs.indist_score = None
    #         outs.pseudo_label = None                       
    #         outs.mh_distance = None                       
    #     return outs    
    


                    


    # def compute_func_samples_z(self, f_mean,f_Lcov ,indist_score, nfsample=5):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
    #     if self.training:
    #         # compute f_samples
    #         if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :           
    #             #z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             #z_samples = z_samples[:,None,:]   #(nb,1,nfsample)                  
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
    #             f_mean_samples = (f_mean[:,:,None]) #(nb,nout,1)
    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             #f_std_samples = f_std_samples*(1 + z_samples)    

    #             f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                        
    #             return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)

    #     else:
    #         # compute f_samples
    #         if not self.usemapestimator:           
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             z_samples = z_samples[:,None,:]                     #(nb,1,nfsample)                  
                
    #             ##(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                                
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples)  ## consider randomness for $z$, but makes generlaization worse                               
    #             f_mean_samples = (f_mean[:,:,None]) 
                
    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)                
    #             #f_std_samples = f_std_samples*(1 + z_samples) ## consider randomness for $z$, but makes generlaization worse                   
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)                

    #             f_samples = f_mean_samples + f_std_samples          #(nb,nout,nfsample)        
    #             return f_samples.permute(2,0,1)                     #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)                                
                    
                    



    # def compute_func_samples_conti_z(self, f_mean,f_Lcov ,indist_score, nfsample=5, datatype='x'):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
        
    #     #if indist_score.dim == 2:
    #     #    

    #     if indist_score.sum().requires_grad:                    
    #         breakpoint()
    #         indist_score_sample= indist_score[:,:,None]
    #         indist_score_samples = torch.cat([F.gumbel_softmax(indist_score,hard=False)[:,:,None] for _ in range(nfsample)],dim=-1)
        
        
    #     if self.training:
    #         # compute f_samples
    #         if not self.usemapestimator and (self.cnt % self.period_stochastic ) == 0 :        
                                
    #             f_mean_samples = f_mean*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
    #             #    breakpoint()
    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #             #f_std_samples = f_std_samples*(1 + indist_score[:,:,None])         
    #             f_std_samples = f_std_samples*(1 + indist_score_samples)         
                           
    #             f_samples = f_mean_samples[:,:,None] + f_std_samples #(nb,nout,nfsample)                        
                                
    #             return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)

    #     else:
    #         # compute f_samples
    #         if not self.usemapestimator:          
    #             #if indist_score.sum().requires_grad:
    #             #    breakpoint()
    #             # z_samples = torch.cat( [ torch.bernoulli(indist_score).detach().clone() for _ in range(nfsample)],dim=-1)
    #             # z_samples = z_samples[:,None,:]                     #(nb,1,nfsample)                  
                
    #             ##(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                                
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - z_samples)  ## consider randomness for $z$, but makes generlaization worse                               
    #             #f_mean_samples = (f_mean[:,:,None]) 
    #             #f_mean_samples = (f_mean[:,:,None])*(1 - indist_score) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)
    #             f_mean_samples = f_mean*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                

    #             eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)                
    #             f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)       
    #             f_std_samples = f_std_samples*(1 + indist_score_samples)    
                        
    #             f_samples = f_mean_samples + f_std_samples          #(nb,nout,nfsample)        
    #             return f_samples.permute(2,0,1)                     #(nfsample,nb,nout)
    #         else:        
    #             return f_mean[None,...] #(1,nb,nout)                               
    


                    
    # ##for fast training
    # def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
                
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute functional dist on train set
    #     q_outs_x           = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')     
    #     logits_sample_x    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        

    #     q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
    #     logits_sample_xc   = self.compute_func_samples_conti_z(q_outs_xc.f_mean, q_outs_xc.f_Lcov, q_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)




    #     #breakpoint()
    #     self.cnt = current_epoch               
    #     #if not self.usemapestimator:            
    #     if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     

    #         # ## set samples from beta distribution
    #         qf_mean = (q_outs_x.f_mean,  q_outs_xc.f_mean)
    #         qf_std  = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
    #         q_indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score) 



    #         # ## set prior distribution on x and mixed xc            
            
    #         #p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')            
    #         p_outs_xc     = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc')            
            
    #         #pf_mean_x_in,   pf_std_x_in    =  p_outs_x.f_mean,    p_outs_x.stdonfs
    #         pf_mean_xc_in,  pf_std_xc_in   =  p_outs_xc.f_mean,   self.priorstd_in*torch.ones_like(p_outs_xc.f_mean).to(xc.device)                 #(nb,nout)
    #         pf_mean_xc_out, pf_std_xc_out  =  torch.zeros_like(pf_mean_xc_in).to(xc.device), self.priorstd_out*torch.ones_like(pf_std_xc_in).to(xc.device)               #(nb,nout) 

    #         beta_samples = self.beta_dist.sample((x.size(0),)).to(x.device) #(nb,1), beta ~ Beta(.1,.1), eventually used as a regression score 
    #         pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples) +  pf_mean_xc_out*beta_samples
    #         pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples)  +  pf_std_xc_out*beta_samples
    #         pf_std_xc_mixed  =  torch.clamp( pf_std_xc_mixed, min=eps_clamp)  #(nb,nout)
            
        

    #         #pf_mean = (pf_mean_x_in, pf_mean_xc_mixed)
    #         #pf_std  = (pf_std_x_in, pf_std_xc_mixed)
    #         p_indistscore = (torch.zeros_like(q_outs_x.indist_score).to(x.device),  beta_samples)



    #         # ## compute sup_kl 
    #         # sup_kl = self.compute_sup_kl( (qf_mean_xc_lat, qf_std_xc_lat), (pf_mean_xc_lat, pf_std_xc_lat), Diag=True, nctset=nctset) 
    #         # sup_kl = self.compute_sup_kl(  (torch.cat(qf_mean,dim=0) , torch.cat(qf_std,dim=0)),  (torch.cat(pf_mean,dim=0) , torch.cat(pf_std,dim=0)),  Diag=True, nctset=nctset) 
    #         sup_kl = self.compute_sup_kl( (q_outs_xc.f_mean,q_outs_xc.stdonfs),  (pf_mean_xc_mixed, pf_std_xc_mixed),  Diag=True, nctset=nctset) 
                        
            
    #         # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
    #         self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
            
            
    #     else :            
    #         # compute stdonfs
    #         #stdfs = (q_outs_x.stdonfs, None)  
    #         qf_std = None                         
    #         # compute indist score            
    #         q_indistscore = (None,None)
    #         p_indistscore = (None,None)
    #         #p_pseudolabel = (None,None)
            
    #         #q_outs_xc = None

    #         #sup_kl = torch.tensor(0.0).to(x.device)      

    #         # l2 regularization on param
    #         param_vec = parameters_to_vector(self.deterministic_nn.parameters())
    #         sup_kl = 0.5 * torch.norm(param_vec, p = 2)

                           
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
    #     outs.logits_xc = logits_sample_xc
        
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     outs.reg   = sup_kl           
    #     outs.stdfs = qf_std   
    #     outs.indistscore = q_indistscore
    #     outs.pseudolabel = p_indistscore
         
    #     return outs                 
    
    
    
    

 



    # #def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x' , append=True   ):
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     ## compute f_mean
    #     h = self.base_nn(x)
    #     h, h_list = self.deterministic_nn_aux(h)        
    #     f_mean = self.func_model(param_q_mean,h)  
        
    #     # compute f_Lcov
    #     jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)


    #     if not self.usemapestimator:            
    #         ## compute indist score to make predicive distribtuion            
    #         outs_mh = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
                        

    #         # compute std on fs 
    #         stdonfs = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
    #         # compute f_Lcov 
    #         f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     

    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = f_Lcov
    #         outs.stdonfs = stdonfs
    #         #outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
    #         outs.h_list = h_list
    #         outs.indist_score = outs_mh.indist_score
    #         outs.pseudo_label = outs_mh.pseudo_label
    #         outs.mh_distance = outs_mh.mh_distance

    #     else:
            
    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = None
    #         outs.stdonfs = None
    #         #outs.stdonfs_z = None
            
    #         outs.h_list = h_list
    #         outs.indist_score = None
    #         outs.pseudo_label = None                       
    #         outs.mh_distance = None                       
    #     return outs    
    



    # def compute_func_samples_conti_z(self, f_mean,f_Lcov ,indist_score, nfsample=5, datatype='x'):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
        
    #     ## indist score
    #     if (not self.usemapestimator):
    #         #assert indist_score is not None
    #         if indist_score is None:
    #             indist_score_samples = torch.zeros(f_mean.shape[0],1,1).to(f_mean.device)                            
    #         else:
    #             indist_prob = torch.cat( [indist_score,1-indist_score],dim=-1)
    #             indist_score_samples = torch.cat( [F.gumbel_softmax(indist_prob, tau=1e0, hard=False,dim=-1)[:,0].view(-1,1,1) for _ in range(nfsample)],dim=-1)   #v63     
    #             #indist_score_samples = torch.cat( [F.gumbel_softmax(indist_prob, tau=1e-1, hard=False,dim=-1)[:,0].view(-1,1,1) for _ in range(nfsample)],dim=-1)                        
                
    #     ## compoute f_samples                
    #     if (not self.usemapestimator) :            
    #         #f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                          ##v61,v62
    #         f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike
    #         eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #         f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)

    #         #f_std_samples = f_std_samples                                   ## v61,v62
    #         f_std_samples = f_std_samples*(1 + indist_score_samples)         ## v63 --> incurs spike
             
    #         f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                        
             
    #         # if indist_prob.sum().requires_grad:                            
    #         #     breakpoint()
                
                
    #         return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #     else:        
    #         return f_mean[None,...] #(1,nb,nout)
                                  




    # ## v64
    # def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
    #     self.setup_functiondist() if not self.flag_setup else 1
        
    #     ## compute f_mean
    #     h = self.base_nn(x)
    #     h, h_list = self.deterministic_nn_aux(h)        
    #     f_mean = self.func_model(param_q_mean,h)  
        
    #     # compute f_Lcov
    #     jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)


    #     if not self.usemapestimator:            
    #         ## compute indist score to make predicive distribtuion            
    #         mh_outs = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
                        

    #         # compute std on fs 
    #         stdonfs = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
    #         # compute f_Lcov 
    #         f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     

    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = f_Lcov
    #         outs.stdonfs = stdonfs
    #         #outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
    #         outs.h_list = h_list
    #         outs.indist_score = mh_outs.indist_score
    #         # outs.pseudo_label = outs_mh.pseudo_label
    #         outs.mh_distance = mh_outs.mh_distance

    #     else:
            
    #         outs = AttrDict()
    #         outs.f_mean = f_mean
    #         outs.f_Lcov = None
    #         outs.stdonfs = None
    #         #outs.stdonfs_z = None
            
    #         outs.h_list = h_list
    #         outs.indist_score = None
    #         # outs.pseudo_label = None                       
    #         outs.mh_distance = None                       
    #     return outs    
    


    # ## v64
    # def compute_func_samples_conti_z(self, f_mean, f_Lcov ,indist_score, nfsample=5, datatype='x'):        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        
        
    #     ## indist score
    #     if (not self.usemapestimator):
    #         #assert indist_score is not None
    #         if indist_score is None:
    #             z_samples = torch.zeros(f_mean.shape[0],1,1).to(f_mean.device)                            
    #         else:
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).unsqueeze(dim=-1) for _ in range(nfsample)],dim=-1)
    #             #if indist_score.sum().requires_grad:
    #             #    breakpoint()
                            
                
    #     ## compoute f_samples                
    #     if (not self.usemapestimator) :            
    #         f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
    #         #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            
    #         eps_samples   = torch.randn(nb,ydim,nfsample).to(f_mean.device)
    #         f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
    #         f_std_samples = f_std_samples*(1 + z_samples)         
    #         #breakpoint()
                         
    #         f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                                        
    #         return f_samples.permute(2,0,1) #(nfsample,nb,nout)
    #     else:        
    #         return f_mean[None,...] #(1,nb,nout)
              

    ## v64
    def compute_func_dist(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        ## compute f_mean
        h = self.base_nn(x)
        h, h_list = self.deterministic_nn_aux(h)        
        f_mean = self.func_model(param_q_mean,h)  
        
        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)


        if not self.usemapestimator:            
            ## compute indist score to make predicive distribtuion            
            mh_outs = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
                        



            # compute std on fs 
            stdonfs = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
            # compute f_Lcov 
            f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)     

            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = f_Lcov
            outs.stdonfs = stdonfs
            #outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
            outs.h_list = h_list
            outs.indist_score = mh_outs.indist_score
            # outs.pseudo_label = outs_mh.pseudo_label
            outs.mh_distance = mh_outs.mh_distance

        else:
            
            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_Lcov = None
            outs.stdonfs = None
            #outs.stdonfs_z = None
            
            outs.h_list = h_list
            outs.indist_score = None
            # outs.pseudo_label = None                       
            outs.mh_distance = None                       
        return outs                                      

    ## v65
    def compute_func_samples_conti_z(self, f_mean, f_Lcov ,indist_score, nfsample=5, datatype='x'):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        
        
        ## indist score
        if (not self.usemapestimator):
            #assert indist_score is not None
            if indist_score is None:
                z_samples = torch.zeros(f_mean.shape[0],1,1).to(f_mean.device)                            
            else:
                z_samples = torch.cat( [ torch.bernoulli(indist_score).unsqueeze(dim=-1) for _ in range(nfsample)],dim=-1)
                #if indist_score.sum().requires_grad:
                #    breakpoint()
                            
                
        ## compoute f_samples                
        if (not self.usemapestimator) :            
            f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
            #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            
            eps_samples   = torch.randn(nb,ydim,nfsample).to(f_mean.device)
            f_std_samples = torch.einsum('bij,bjk->bik',f_Lcov,eps_samples)
            #f_std_samples = f_std_samples*(1 + z_samples)         
            f_std_samples = f_std_samples*z_samples         
            
            # if indist_score.sum().requires_grad:
            #     breakpoint()
                         
            f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                                        
            return f_samples.permute(2,0,1) #(nfsample,nb,nout)
        else:        
            return f_mean[None,...] #(1,nb,nout)
                          
                          
                          
                          
    ## v70
    def compute_func_dist2(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        ## compute f_mean
        h = self.base_nn(x)
        h, h_list = self.deterministic_nn_aux(h)        
        f_mean = self.func_model(param_q_mean,h)  
        
        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)

        if not self.usemapestimator:            
            ## compute indist score to make predicive distribtuion            
            mh_outs = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )                                      

            # compute std on fs 
            f_ustd = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
            
            # compute f_cov diag in approximate way             
            #f_left_cov2 = U*f_ustd[:,None,:]
            #f_cov2 = f_left_cov2 @ f_left_cov2.permute(0,2,1)
            #f_std2 = torch.diagonal(f_cov2,dim1=-1, dim2=-2).sqrt()

            # compute f_cov diag in exact way  
            param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
            param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
            f_left_cov = jacobian_f_vec*param_q_std_vec
            f_cov  = f_left_cov @ f_left_cov.permute(0,2,1)
            f_std = torch.diagonal(f_cov,dim1=-1, dim2=-2).sqrt()
            #breakpoint()
            

            outs = AttrDict()
            outs.f_mean = f_mean
            #outs.f_Lcov = U*(stdonfs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)                 
            outs.f_u = U
            #outs.stdonfs = stdonfs
            outs.f_ustd = f_ustd
            outs.f_std = f_std            
            #outs.stdonfs_z = stdonfs*(1 +  outs_mh.indist_score.detach().clone())            
            outs.h_list = h_list
            outs.indist_score = mh_outs.indist_score
            # outs.pseudo_label = outs_mh.pseudo_label
            outs.mh_distance = mh_outs.mh_distance

        else:
            
            outs = AttrDict()
            outs.f_mean = f_mean
            # outs.f_Lcov = None
            outs.f_u = None            
            outs.f_ustd = None
            outs.f_std = None
            
            outs.h_list = h_list
            outs.indist_score = None
            # outs.pseudo_label = None                       
            outs.mh_distance = None                       
        return outs    
                              
                              
    ## v70
    def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x'):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        
        
        ## indist score
        if (not self.usemapestimator):
            #assert indist_score is not None
            if indist_score is None:
                z_samples = torch.zeros(f_mean.shape[0],1,1).to(f_mean.device)                            
            else:
                z_samples = torch.cat( [ torch.bernoulli(indist_score).unsqueeze(dim=-1) for _ in range(nfsample)],dim=-1)
                #if indist_score.sum().requires_grad:
                #    breakpoint()
                
        
        ## compoute f_samples                
        if (not self.usemapestimator) :            
            
            #print('datatype: {}, indist_score:{}, z_samples:{}'.format(datatype, indist_score,z_samples))
            #print('\n')
            
            f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
            #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            
            eps_samples = torch.randn(nb,ydim,nfsample).to(f_mean.device)            
            eps_samples = eps_samples*f_ustd[:,:,None]                        #(nb,nout,nfsample)
            eps_z_samples = eps_samples*z_samples 
            f_std_samples = torch.einsum('bij,bjk->bik',f_u,eps_z_samples)
            #if indist_score.sum().requires_grad:
            #    breakpoint()
                        
            f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                                        
            return f_samples.permute(2,0,1) #(nfsample,nb,nout)
        else:        
            return f_mean[None,...] #(1,nb,nout)
                                                        


    # ##for fast training
    # ## predictive entrophy + new regularization term
    # def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
                
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute functional dist on train set
    #     q_outs_x           = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')     
    #     logits_sample_x    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        

    #     #breakpoint()
    #     self.cnt = current_epoch               
    #     #if not self.usemapestimator:            
    #     if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     
    #         q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
    #         logits_sample_xc   = self.compute_func_samples_conti_z(q_outs_xc.f_mean, q_outs_xc.f_Lcov, q_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)

    #         # # ## set samples from beta distribution
    #         # #qf_mean = (q_outs_x.f_mean,  q_outs_xc.f_mean)
    #         qf_std       = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
    #         q_indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score)             

    #         entropy_samples, rescaled_entropy_samples = self.get_entropy(logits_sample_xc)
    #         beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)
    #         #beta_samples = torch.bernoulli(beta_samples)
    #         beta_samples_hard = torch.where(beta_samples>=0.5,
    #                                         torch.ones_like(beta_samples).to(x.device),
    #                                         torch.zeros_like(beta_samples).to(x.device) )
            
            
    #         q_mhdistance = (q_outs_x.mh_distance,q_outs_xc.mh_distance)

    #         # ## set prior distribution on x and mixed xc            
            
    #         #p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')            
    #         p_outs_xc     = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc')            
            
    #         #pf_mean_x_in,   pf_std_x_in    =  p_outs_x.f_mean,    p_outs_x.stdonfs
    #         pf_mean_xc_in,  pf_std_xc_in   =  p_outs_xc.f_mean,                              self.priorstd_in*torch.ones_like(p_outs_xc.f_mean).to(xc.device)                 #(nb,nout)
    #         pf_mean_xc_out, pf_std_xc_out  =  torch.zeros_like(pf_mean_xc_in).to(xc.device), self.priorstd_out*torch.ones_like(pf_std_xc_in).to(xc.device)               #(nb,nout) 

    #         #beta_samples = self.beta_dist.sample((x.size(0),)).to(x.device) #(nb,1), beta ~ Beta(.1,.1), eventually used as a regression score 
    #         # pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples) +  pf_mean_xc_out*beta_samples
    #         # pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples)  +  pf_std_xc_out*beta_samples
    #         pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples_hard) +  pf_mean_xc_out*beta_samples_hard
    #         pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples_hard)  +  pf_std_xc_out*beta_samples_hard

    #         pf_std_xc_mixed  =  torch.clamp( pf_std_xc_mixed, min=eps_clamp)  #(nb,nout)
            
            

    #         #pf_mean = (pf_mean_x_in, pf_mean_xc_mixed)
    #         #pf_std  = (pf_std_x_in, pf_std_xc_mixed)
    #         p_indistscore = (torch.zeros_like(q_outs_x.indist_score).to(x.device),  beta_samples)



    #         # ## compute sup_kl 
    #         # sup_kl = self.compute_sup_kl( (qf_mean_xc_lat, qf_std_xc_lat), (pf_mean_xc_lat, pf_std_xc_lat), Diag=True, nctset=nctset) 
    #         # sup_kl = self.compute_sup_kl(  (torch.cat(qf_mean,dim=0) , torch.cat(qf_std,dim=0)),  (torch.cat(pf_mean,dim=0) , torch.cat(pf_std,dim=0)),  Diag=True, nctset=nctset) 
    #         sup_kl = self.compute_sup_kl( (q_outs_xc.f_mean, q_outs_xc.stdonfs),  (pf_mean_xc_mixed, pf_std_xc_mixed),  Diag=True, nctset=nctset) 
                        
            
    #         # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
    #         self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
    #         #breakpoint()
            
    #     else :            
    #         # compute stdonfs
    #         #stdfs = (q_outs_x.stdonfs, None)  
    #         logits_sample_xc = None
    #         qf_std = None                         
    #         # compute indist score     
    #         q_mhdistance = (None,None)                   
    #         q_indistscore = (None,None)
    #         p_indistscore = (None,None)
    #         #p_pseudolabel = (None,None)
            
    #         #q_outs_xc = None

    #         #sup_kl = torch.tensor(0.0).to(x.device)      

    #         # l2 regularization on param
    #         param_vec = parameters_to_vector(self.deterministic_nn.parameters())
    #         sup_kl = 0.5 * torch.norm(param_vec, p = 2)

                           
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
    #     outs.logits_xc = logits_sample_xc
        
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     #outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     outs.reg   = sup_kl           
    #     outs.qfstd = qf_std   
    #     outs.indistscore = q_indistscore
    #     outs.mhdistance = q_mhdistance   
    #     outs.pseudolabel = p_indistscore
         
    #     return outs                     
    




    # ##for fast training
    # ## predictive entrophy + new regularization term
    # def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
                
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute functional dist on train set
    #     q_outs_x           = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')     
    #     logits_sample_x    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        
    #     q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
    #     logits_sample_xc   = self.compute_func_samples_conti_z(q_outs_xc.f_mean, q_outs_xc.f_Lcov, q_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)

    #     # # ## set samples from beta distribution
    #     # #qf_mean = (q_outs_x.f_mean,  q_outs_xc.f_mean)
    #     qf_std       = (q_outs_x.stdonfs, q_outs_xc.stdonfs)                   
    #     q_indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score)             
    #     q_mhdistance = (q_outs_x.mh_distance,q_outs_xc.mh_distance)


    #     #breakpoint()
    #     self.cnt = current_epoch               
    #     #if not self.usemapestimator:            
    #     if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     

    #         entropy_samples, rescaled_entropy_samples = self.get_entropy(logits_sample_xc)
    #         beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)
    #         #beta_samples = torch.bernoulli(beta_samples)
    #         beta_samples_hard = torch.where(beta_samples>=0.5,                                ##v68
    #                                         torch.ones_like(beta_samples).to(x.device),
    #                                         torch.zeros_like(beta_samples).to(x.device) )
            
            

    #         # ## set prior distribution on x and mixed xc            
            
    #         #p_outs_x      = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')            
    #         p_outs_xc     = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc')            
            
    #         #pf_mean_x_in,   pf_std_x_in    =  p_outs_x.f_mean,    p_outs_x.stdonfs
    #         pf_mean_xc_in,  pf_std_xc_in   =  p_outs_xc.f_mean,                              self.priorstd_in*torch.ones_like(p_outs_xc.f_mean).to(xc.device)                 #(nb,nout)
    #         pf_mean_xc_out, pf_std_xc_out  =  torch.zeros_like(pf_mean_xc_in).to(xc.device), self.priorstd_out*torch.ones_like(pf_std_xc_in).to(xc.device)               #(nb,nout) 

    #         #beta_samples = self.beta_dist.sample((x.size(0),)).to(x.device) #(nb,1), beta ~ Beta(.1,.1), eventually used as a regression score 
    #         # pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples) +  pf_mean_xc_out*beta_samples
    #         # pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples)  +  pf_std_xc_out*beta_samples
    #         pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples_hard) +  pf_mean_xc_out*beta_samples_hard
    #         pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples_hard)  +  pf_std_xc_out*beta_samples_hard

    #         pf_std_xc_mixed  =  torch.clamp( pf_std_xc_mixed, min=eps_clamp)  #(nb,nout)
            
            

    #         #pf_mean = (pf_mean_x_in, pf_mean_xc_mixed)
    #         #pf_std  = (pf_std_x_in, pf_std_xc_mixed)
    #         p_indistscore = (torch.zeros_like(q_outs_x.indist_score).to(x.device),  beta_samples)



    #         # ## compute sup_kl 
    #         # sup_kl = self.compute_sup_kl( (qf_mean_xc_lat, qf_std_xc_lat), (pf_mean_xc_lat, pf_std_xc_lat), Diag=True, nctset=nctset) 
    #         # sup_kl = self.compute_sup_kl(  (torch.cat(qf_mean,dim=0) , torch.cat(qf_std,dim=0)),  (torch.cat(pf_mean,dim=0) , torch.cat(pf_std,dim=0)),  Diag=True, nctset=nctset) 
    #         sup_kl = self.compute_sup_kl( (q_outs_xc.f_mean, q_outs_xc.stdonfs),  (pf_mean_xc_mixed, pf_std_xc_mixed),  Diag=True, nctset=nctset) 
                        
            
    #         # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
    #         self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
    #         #breakpoint()
            
    #     else :            
    #         # compute stdonfs
    #         #stdfs = (q_outs_x.stdonfs, None)  
    #         #logits_sample_xc = None
    #         #qf_std = None             
    #         #q_indistscore = (None,None)            
                        
    #         # compute indist score     
    #         #q_mhdistance = (None,None)                   
    #         p_indistscore = (None,None)
    #         #p_pseudolabel = (None,None)
            
    #         #q_outs_xc = None

    #         #sup_kl = torch.tensor(0.0).to(x.device)      

    #         # l2 regularization on param
    #         param_vec = parameters_to_vector(self.deterministic_nn.parameters())
    #         sup_kl = 0.5 * torch.norm(param_vec, p = 2)

                           
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
    #     outs.logits_xc = logits_sample_xc
        
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     #outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     outs.reg   = sup_kl           
    #     outs.qfstd = qf_std   
    #     outs.indistscore = q_indistscore
    #     outs.mhdistance = q_mhdistance   
    #     outs.pseudolabel = p_indistscore
         
    #     return outs                     
    
    


    ##for fast training
    ## predictive entrophy + new regularization term
    ## v68
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
        # q_outs_x           = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')     
        # logits_sample_x    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        # logits_sample_x    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_std, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        logits_sample_x    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        
        self.cnt = current_epoch               
        #if not self.usemapestimator:            
        if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     
            
            
            # ## set prior distribution on x and mixed xc                        
            # p_outs_xc     = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc')            
            # logits_sample_xc   = self.compute_func_samples_conti_z(p_outs_xc.f_mean, p_outs_xc.f_Lcov, p_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)
            p_outs_xc     = self.compute_func_dist2( xc, param_p_mean, param_p_logstd, datatype='xc')            
            #logits_sample_xc   = self.compute_func_samples_conti_z2(p_outs_xc.f_mean, p_outs_xc.f_u, p_outs_xc.f_std, p_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)
            logits_sample_xc   = self.compute_func_samples_conti_z2(p_outs_xc.f_mean, p_outs_xc.f_u, p_outs_xc.f_ustd, p_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)
            
            entropy_samples, rescaled_entropy_samples = self.get_entropy(logits_sample_xc)
            beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)
            #beta_samples = torch.bernoulli(beta_samples)
            beta_samples_hard = torch.where(beta_samples>=0.5,                                ##v68
                                            torch.ones_like(beta_samples).to(x.device),
                                            torch.zeros_like(beta_samples).to(x.device) )
                        

            #pf_mean_x_in,   pf_std_x_in    =  p_outs_x.f_mean,    p_outs_x.stdonfs
            pf_mean_xc_in,  pf_std_xc_in   =  p_outs_xc.f_mean,                              self.priorstd_in*torch.ones_like(p_outs_xc.f_mean).to(xc.device)                 #(nb,nout)
            pf_mean_xc_out, pf_std_xc_out  =  torch.zeros_like(pf_mean_xc_in).to(xc.device), self.priorstd_out*torch.ones_like(pf_std_xc_in).to(xc.device)               #(nb,nout) 

            #beta_samples = self.beta_dist.sample((x.size(0),)).to(x.device) #(nb,1), beta ~ Beta(.1,.1), eventually used as a regression score 
            # pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples) +  pf_mean_xc_out*beta_samples
            # pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples)  +  pf_std_xc_out*beta_samples
            pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples_hard) +  pf_mean_xc_out*beta_samples_hard
            pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples_hard)  +  pf_std_xc_out*beta_samples_hard

            pf_std_xc_mixed  =  torch.clamp( pf_std_xc_mixed, min=eps_clamp)  #(nb,nout)
            
            
            # if q_outs_x.indist_score.sum().requires_grad :
            #     breakpoint()


            ## set variational distribution on xc
            # q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
            q_outs_xc          = self.compute_func_dist2( xc, param_q_mean, param_q_logstd, datatype='xc')     

            # ## compute sup_kl 
            # sup_kl = self.compute_sup_kl(  (torch.cat(qf_mean,dim=0) , torch.cat(qf_std,dim=0)),  (torch.cat(pf_mean,dim=0) , torch.cat(pf_std,dim=0)),  Diag=True, nctset=nctset) 
            sup_kl = self.compute_sup_kl( (q_outs_xc.f_mean, q_outs_xc.f_std),  (pf_mean_xc_mixed, pf_std_xc_mixed),  Diag=True, nctset=nctset) 
            
            #breakpoint()
            
            # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
            self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
            #breakpoint()
            
            

            qf_std       = (q_outs_x.f_std, q_outs_xc.f_std)                   
            q_indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score)             
            q_mhdistance = (q_outs_x.mh_distance,q_outs_xc.mh_distance)            
            p_indistscore = (torch.zeros_like(q_outs_x.indist_score).to(x.device),  beta_samples)
            
        else :            
            # compute stdonfs
            #stdfs = (q_outs_x.stdonfs, None)  
            #logits_sample_xc = None
                        
            # compute indist score     
            #p_pseudolabel = (None,None)
            
            #q_outs_xc = None

            #sup_kl = torch.tensor(0.0).to(x.device)      

            # l2 regularization on param
            param_vec = parameters_to_vector(self.deterministic_nn.parameters())
            sup_kl = 0.5 * torch.norm(param_vec, p = 2)
            
            logits_sample_xc = None
            qf_std = (None,None)            
            q_indistscore = (None,None)            
            q_mhdistance = (None,None)                                           
            p_indistscore = (None,None)
            

                           
        # outputs        
        outs = AttrDict()
        outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
        outs.logits_xc = logits_sample_xc
        
        #outs.log_qf,outs.log_pf = log_qf, log_pf
        #outs.log_qf,outs.log_pf = None,None
        #outs.sup_kl = sup_kl                
        outs.reg   = sup_kl           
        outs.qfstd = qf_std   
        outs.indistscore = q_indistscore
        outs.mhdistance = q_mhdistance   
        outs.pseudolabel = p_indistscore
         
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



    def get_entropy(self, logits,  scale=2.3):    
        #misclassification_targets = (1-misclass_tgt(preds, targets, topk)).astype(bool)

        #if criterion == 'entropy':
        #pred_samples     = F.softmax(logits, dim=-1).detach().clone()  #(nfsample, nb, nout)
        pred_samples     = F.softmax(logits / self.betahyp, dim=-1).detach().clone()  #(nfsample, nb, nout)
        
        # if logits.sum().requires_grad:
        #     breakpoint()
        
        log_pred_samples = torch.log(pred_samples + eps_log)      #(nfsample, nb, nout)        
        entropy_samples  = -torch.sum(pred_samples*log_pred_samples,dim=-1)  #(nfsample, nb)
        rescaled_entropy_samples = torch.clamp(entropy_samples /scale , min=0.0, max=1.0)  #(nfsample, nb)
        return entropy_samples, rescaled_entropy_samples
    
        #entropy = np.sum(-(logits + eps_log ) * np.log(preds + eps_log ), axis=1)





    # def predict_eval(self, x , nfsample=1):        
    #     #print('nfsample: {} in eval'.format(nfsample))        
    #     self.eval()        
    #     # set params
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        
    #     #logits_sample = self.compute_func_samples(q_outs_x.f_mean,outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
    #     # logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
    #     logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)


    #     rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
    #     beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


    #     pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)               
    #     outs.logits = logits_sample
    #     outs.stdfs = q_outs_x.stdonfs
    #     outs.indist_score = q_outs_x.indist_score
    #     outs.mh_distance = q_outs_x.mh_distance 
    #     outs.pseudo_label = beta_samples
        
    #     return outs                 
                                                           
        
        
        


    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_std, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_mean, p_outs_x.f_u, p_outs_x.f_std, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        #outs.stdfs = q_outs_x.stdonfs
        outs.indist_score = q_outs_x.indist_score
        outs.mh_distance = q_outs_x.mh_distance 
        outs.pseudo_label = beta_samples
        outs.pseudo_labelp = beta_samples_p        
        return outs                 
                                                           


    def predict_eval_oracle(self, x , nfsample=1 ,datatype = 'x'):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')

        if datatype == 'x':
            oracle_indist_score = torch.zeros_like(q_outs_x.indist_score).to(x.device)
        if datatype == 'xo':
            oracle_indist_score = torch.ones_like(q_outs_x.indist_score).to(x.device)             
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_std, oracle_indist_score, nfsample=nfsample, datatype=datatype) #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        # rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        # beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        # p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_ mean, p_outs_x.f_u, p_outs_x.f_std, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        # rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        # beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        #outs.stdfs = q_outs_x.stdonfs
        # outs.indist_score = q_outs_x.indist_score
        # outs.mh_distance = q_outs_x.mh_distance 
        # outs.pseudo_label = beta_samples
        # outs.pseudo_labelp = beta_samples_p        
        return outs                 
        








        








##---------------------------------------------------------------------------------------------------------
## FBNN_RPAwareclscatsimplex_Base
##---------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
#from torch.distributions.continuous_bernoulli import ContinuousBernoulli as CBERN

class FBNN_RPContLatAwaSimp_Base(FBNN_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10, losstype='elbo'):
    def __init__(self, priorstd_in=1e-1, priorstd_out=5e-1, scalelogstd = -8, losstype='elbo', updateperiod=1 ,betahyp=.1):
                        
        super(FBNN_RPContLatAwaSimp_Base, self).__init__( scalelogstd, losstype)

        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

        self.priorstd_in  =  priorstd_in               
        self.priorstd_out =  priorstd_out       
        
                
        self.base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None
        self.static_mh = None

        
        self.usemapestimator = False
        self.setprior=False
        
        self.cnt = 0       
        self.period_stochastic = updateperiod
        #self.period_stochastic = 1        
        #self.period_stochastic = 2
        #self.period_stochastic = 3

        self.betahyp = betahyp
        #self.beta_dist = Beta(torch.tensor([betahyp]),torch.tensor([betahyp]))
        
        
        
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn) ##probablistc simplex as output                
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         
        
        
    def setup_parameters(self):        
        #self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )        
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

        return   
            

    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                
            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)
    
    
    def base_nn_aux(self, x):
        return self.base_nn(x)
    

    def deterministic_nn_aux(self, h):
        h_feature_list = []
        for ith_module in self.deterministic_nn:
            h = ith_module(h)
            h_feature_list.append(h.view(h.size(0),h.size(1),-1 ).mean(dim=-1) )
        return h, h_feature_list
    


    def _compute_stdonfs(self, param_q_logstd, U, S, VT):
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
        sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
        eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
        stdonfs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)                                    
        return stdonfs
                          
                          
    ## v70
    def compute_func_dist2(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        ## compute f_mean
        h = self.base_nn(x)
        h, h_list = self.deterministic_nn_aux(h)        
        f_mean = self.func_model(param_q_mean,h)  
        
        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
                        #for k in range
            rank = torch.where( S.min(dim=0)[0] > 1e-8 )[0][-1].item() + 1
            U, S, VT = U[:,:,:rank] , S[:,:rank], VT[:,:rank,:]  
        

        if not self.usemapestimator:            
            ## compute indist score to make predicive distribtuion            
            mh_outs = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
 
 
                        
            # compute std on fs 
            f_ustd = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
            
            f_left_cov = U*f_ustd[:,None,:]
            f_cov = f_left_cov @ f_left_cov.permute(0,2,1)
            f_std = torch.diagonal(f_cov,dim1=-1, dim2=-2).sqrt()
            # compute f_Lcov 

            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_u = U
            outs.f_ustd = f_ustd 
            outs.f_std = f_std
            outs.h_list = h_list
            outs.indist_score = mh_outs.indist_score
            outs.mh_distance = mh_outs.mh_distance

        else:
            
            outs = AttrDict()
            outs.f_mean = f_mean
            outs.f_u = None    
            outs.f_ustd = None        
            outs.f_std = None
            outs.h_list = h_list
            outs.indist_score = None
            outs.mh_distance = None                       
        return outs    
                              
                              
    ## v70
    def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x'):        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        #nb,ydim = f_mean.shape
        
        
        ## indist score
        if (not self.usemapestimator):
            #assert indist_score is not None
            if indist_score is None:
                z_samples = torch.zeros(f_mean.shape[0],1,1).to(f_mean.device)                            
            else:
                z_samples = torch.cat( [ torch.bernoulli(indist_score).unsqueeze(dim=-1) for _ in range(nfsample)],dim=-1)
                #if indist_score.sum().requires_grad:
                #    breakpoint()
                

        ## compoute f_samples                
        if (not self.usemapestimator) :            
            
            #print('datatype: {}, indist_score:{}, z_samples:{}'.format(datatype, indist_score,z_samples))
            #print('\n')
            
            f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
            #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            

            nb,rank = f_ustd.shape
            eps_samples = torch.randn(nb,rank,nfsample).to(f_mean.device)            
                        
            eps_samples = eps_samples*f_ustd[:,:,None]                        #(nb,nout,nfsample)
            eps_z_samples = eps_samples*z_samples 
            f_std_samples = torch.einsum('bij,bjk->bik',f_u,eps_z_samples)
            #if indist_score.sum().requires_grad:
            #    breakpoint()
                        
            f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                    
            f_samples = f_samples.permute(2,0,1)     
            if not self.usemapestimator and indist_score.sum().requires_grad : 
                breakpoint()
            
            f_samples = F.softmax(f_samples,dim=-1) #(nfsample,nb,nout)                    
            return f_samples
        else:        
            return f_mean[None,...] #(1,nb,nout)
                                                        

    


    ##for fast training
    ## predictive entrophy + new regularization term
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
                
        q_outs_x          = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        q_pred_sample_x   = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        logits_sample_x = torch.log(q_pred_sample_x  + eps_log) 
        #breakpoint()
        
        
        self.cnt = current_epoch               
        #if not self.usemapestimator:            
        if (not self.usemapestimator) and (current_epoch % self.period_stochastic ) == 0 :     
            
            ## set variational distribution on xc
            # q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
            q_outs_xc          = self.compute_func_dist2( xc, param_q_mean, param_q_logstd, datatype='xc')     
            
            
            # ## set prior distribution on x and mixed xc                        
            # p_outs_xc     = self.compute_func_dist( xc, param_p_mean, param_p_logstd, datatype='xc')            
            # logits_sample_xc   = self.compute_func_samples_conti_z(p_outs_xc.f_mean, p_outs_xc.f_Lcov, p_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)
            p_outs_xc          = self.compute_func_dist2( xc, param_p_mean, param_p_logstd, datatype='xc')            
            p_pred_sample_xc   = self.compute_func_samples_conti_z2(p_outs_xc.f_mean, p_outs_xc.f_u, p_outs_xc.f_ustd, p_outs_xc.indist_score, nfsample=nfsample, datatype='xc') #(nfsample,nb,nout)
            logits_sample_xc   = torch.log(p_pred_sample_xc + eps_log)            
            
            _, rescaled_entropy_samples = self.get_entropy(p_pred_sample_xc)
            beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)
            #beta_samples = torch.bernoulli(beta_samples)
            beta_samples_hard = torch.where(beta_samples>=0.5,                                ##v68
                                            torch.ones_like(beta_samples).to(x.device),
                                            torch.zeros_like(beta_samples).to(x.device) )
                        

            ## on probablistic space, instead of logits space 
            pf_mean_xc_in,  pf_std_xc_in   =  p_outs_xc.f_mean,                                                self.priorstd_in*torch.ones_like(p_outs_xc.f_mean).to(xc.device)                 #(nb,nout)
            pf_mean_xc_out, pf_std_xc_out  =  F.softmax(torch.ones_like(pf_mean_xc_in).to(xc.device),dim=-1),  self.priorstd_out*torch.ones_like(pf_std_xc_in).to(xc.device)               #(nb,nout) 
            pf_mean_xc_mixed =  pf_mean_xc_in*(1 - beta_samples_hard) +  pf_mean_xc_out*beta_samples_hard
            pf_std_xc_mixed  =  pf_std_xc_in*(1 - beta_samples_hard)  +  pf_std_xc_out*beta_samples_hard
            pf_std_xc_mixed  =  torch.clamp( pf_std_xc_mixed, min=eps_clamp)  #(nb,nout)
            
            
            if not self.usemapestimator and q_outs_x.indist_score.sum().requires_grad : 
                breakpoint()


            # ## compute sup_kl 
            # sup_kl = self.compute_sup_kl( (qf_mean_xc_lat, qf_std_xc_lat), (pf_mean_xc_lat, pf_std_xc_lat), Diag=True, nctset=nctset) 
            # sup_kl = self.compute_sup_kl(  (torch.cat(qf_mean,dim=0) , torch.cat(qf_std,dim=0)),  (torch.cat(pf_mean,dim=0) , torch.cat(pf_std,dim=0)),  Diag=True, nctset=nctset) 
            sup_kl = self.compute_sup_kl( (q_outs_xc.f_mean, q_outs_xc.f_std) ,  (pf_mean_xc_mixed, pf_std_xc_mixed),  Diag=True, nctset=nctset) 
                        
            
            # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
            self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
            
            #breakpoint()
            qf_std       = (q_outs_x.f_std, q_outs_xc.f_std)                   
            q_indistscore = (q_outs_x.indist_score,q_outs_xc.indist_score)             
            q_mhdistance = (q_outs_x.mh_distance,q_outs_xc.mh_distance)            
            p_indistscore = (torch.zeros_like(q_outs_x.indist_score).to(x.device),  beta_samples)
            
            
        else :            
            # l2 regularization on param
            param_vec = parameters_to_vector(self.deterministic_nn.parameters())
            sup_kl = 0.5 * torch.norm(param_vec, p = 2)
            
            logits_sample_xc = None
            qf_std = (None,None)            
            q_indistscore = (None,None)            
            q_mhdistance = (None,None)                                           
            p_indistscore = (None,None)
            

                           
        # outputs        
        outs = AttrDict()
        outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
        outs.logits_xc = logits_sample_xc 
        #outs.logits    = torch.log(q_pred_sample_x  + eps_log)                #logits_sample_x,logits_sample_xc
        #outs.logits_xc = torch.log(p_pred_sample_xc + eps_log)
        
        #outs.log_qf,outs.log_pf = log_qf, log_pf
        #outs.log_qf,outs.log_pf = None,None
        #outs.sup_kl = sup_kl                
        outs.reg   = sup_kl           
        outs.qfstd = qf_std   
        outs.indistscore = q_indistscore
        outs.mhdistance = q_mhdistance   
        outs.pseudolabel = p_indistscore
         
        return outs                     
    


    def get_entropy(self, logit_samples,  scale=2.3):    
    #def get_entropy(self, pred_samples,  scale=2.3):
        
        pred_samples     = F.softmax(logit_samples, dim=-1).detach().clone()  #(nfsample, nb, nout)
            
        pred_samples     = pred_samples.detach().clone()  #(nfsample, nb, nout)
        log_pred_samples = torch.log(pred_samples + eps_log)      #(nfsample, nb, nout)        
        entropy_samples  = -torch.sum(pred_samples*log_pred_samples,dim=-1)  #(nfsample, nb)
        rescaled_entropy_samples = torch.clamp(entropy_samples /scale , min=0.0, max=1.0)  #(nfsample, nb)
        return entropy_samples, rescaled_entropy_samples





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



    
        #entropy = np.sum(-(logits + eps_log ) * np.log(preds + eps_log ), axis=1)





    # def predict_eval(self, x , nfsample=1):        
    #     #print('nfsample: {} in eval'.format(nfsample))        
    #     self.eval()        
    #     # set params
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        
    #     #logits_sample = self.compute_func_samples(q_outs_x.f_mean,outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
    #     # logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
    #     logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)


    #     rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
    #     beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


    #     pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)               
    #     outs.logits = logits_sample
    #     outs.stdfs = q_outs_x.stdonfs
    #     outs.indist_score = q_outs_x.indist_score
    #     outs.mh_distance = q_outs_x.mh_distance 
    #     outs.pseudo_label = beta_samples
        
    #     return outs                 
                                                           
        
        
        


    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_std, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_mean, p_outs_x.f_u, p_outs_x.f_std, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        #outs.stdfs = q_outs_x.stdonfs
        outs.indist_score = q_outs_x.indist_score
        outs.mh_distance = q_outs_x.mh_distance 
        outs.pseudo_label = beta_samples
        outs.pseudo_labelp = beta_samples_p        
        return outs                 
                                                           


    def predict_eval_oracle(self, x , nfsample=1 ,datatype = 'x'):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')

        if datatype == 'x':
            oracle_indist_score = torch.zeros_like(q_outs_x.indist_score).to(x.device)
        if datatype == 'xo':
            oracle_indist_score = torch.ones_like(q_outs_x.indist_score).to(x.device)             
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_std, oracle_indist_score, nfsample=nfsample, datatype=datatype) #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        # rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        # beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        # p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_ mean, p_outs_x.f_u, p_outs_x.f_std, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        # rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        # beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        #outs.stdfs = q_outs_x.stdonfs
        # outs.indist_score = q_outs_x.indist_score
        # outs.mh_distance = q_outs_x.mh_distance 
        # outs.pseudo_label = beta_samples
        # outs.pseudo_labelp = beta_samples_p        
        return outs                 
        
        
        
        
        
        
        
        
        
        
        
        
        







##---------------------------------------------------------------------------------------------------------
## FBNN_RPAwareclscatsimplex_Base
##---------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
#from torch.distributions.continuous_bernoulli import ContinuousBernoulli as CBERN

class FBNN_RPCDLat_Base(FBNN_Base):
    #def __init__(self, block, num_blocks, num_classes=10, scalelogstd = -8, num_split_lastlayers = 0):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10):
    #def __init__(self, block, num_blocks, in_dim=3, out_dim=10, scalelogstd = -8, mh_quantile=0.05, mh_nepochprior=10, losstype='elbo'):
    def __init__(self, priorstd_in=1e-1, priorstd_out=5e-1, scalelogstd = -8, losstype='elbo', updateperiod=1 ,betahyp=.1):
                        
        super(FBNN_RPCDLat_Base, self).__init__( scalelogstd, losstype)

        self.scalelogstd = scalelogstd        
        self.flag_setup = False
        self.losstype = losstype # elbo, iwae, sumo

        self.priorstd_in  =  priorstd_in               
        self.priorstd_out =  priorstd_out       
        
                
        self.base_nn = None
        self.deterministic_nn = None
        self.stochastic_nn = None
        self.static_mh = None

        
        self.usemapestimator = False
        self.setprior=False
        
        self.cnt = 0       
        self.period_stochastic = updateperiod
        self.betahyp = betahyp
        self.f_priorstd = 5.
        
        
    def setup_functiondist(self):    
        func_model, _ , buffers = make_functional_with_buffers(self.stochastic_nn) ##probablistc simplex as output --> does not work     
        #self.func_model = func_model        
        self.func_model =  lambda params, x_in : func_model(params, buffers, x_in) 
        self.flag_setup = True         
        return         
        
        
    def setup_parameters(self):        
        #self.num_params_d = _num_parameters(self.deterministic_nn)          
        self.num_params_s = _num_parameters(self.stochastic_nn)  
        self.params_q_logstd =  nn.ParameterList([nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.stochastic_nn.parameters()])
         
        #set prior params        
        self.params_p_mean = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()] )        
        self.params_p_logstd = nn.ParameterList([nn.Parameter(torch.ones_like(ith_param) ,requires_grad=False )  for ith_param in self.stochastic_nn.parameters()]) 

        return   
            

    def compute_jacobian_f_vec(self,q_params, x):        
        with torch.no_grad():
            jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                
            
        jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
        return jacobian_f_vec #(nb,ydim,nparams)
    
    
    def base_nn_aux(self, x):
        return self.base_nn(x)
    

    def deterministic_nn_aux(self, h):
        h_feature_list = []
        for ith_module in self.deterministic_nn:
            h = ith_module(h)
            h_feature_list.append(h.view(h.size(0),h.size(1),-1 ).mean(dim=-1) )
        return h, h_feature_list
    


    # def _compute_stdonfs(self, param_q_logstd, U, S, VT):
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
    #     sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
    #     eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
    #     stdonfs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)                                    
    #     return stdonfs
                          
    def _compute_stdonfs(self, param_q_logstd, U, S, VT):
        param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
        param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        

        ## debugging refelected 
        VT = VT.transpose(-2,-1) #(nb,nothgonalvetors,nparams)                 
        sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nvectors)
        eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
        stdonfs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)                                    
        return stdonfs

                          
    ## v70
    def compute_func_dist2(self, x , param_q_mean, param_q_logstd,   datatype='x'    ):
        
        self.setup_functiondist() if not self.flag_setup else 1
        
        ## compute f_mean
        h = self.base_nn(x)
        h, h_list = self.deterministic_nn_aux(h)        
        f_mean = self.func_model(param_q_mean,h)  
        
        # compute f_Lcov
        jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, h)                
        with torch.no_grad():        
            U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
        

        mh_outs = self.static_mh.compute_indist_score_batch( h_list , datatype = datatype )              
        # compute std on fs 
        f_ustd = self._compute_stdonfs(param_q_logstd, U, S, VT)                        
        
        #f_left_cov = U*f_ustd[:,None,:]
        #f_cov = f_left_cov @ f_left_cov.permute(0,2,1)
        #f_std = torch.diagonal(f_cov,dim1=-1, dim2=-2).sqrt()

        outs = AttrDict()
        outs.f_mean = f_mean
        outs.f_u = U
        outs.f_ustd = f_ustd 
        outs.f_std = (1-mh_outs.indist_score)*f_ustd + mh_outs.indist_score*self.f_priorstd
        outs.h_list = h_list
        outs.indist_score = mh_outs.indist_score
        outs.mh_distance = mh_outs.mh_distance

        return outs    


                              
    # #def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x'):        
    # def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x', upper_bound = 10):        
        
    #     #f_mean,f_Lcov = self.compute_dist_fs(x)        
    #     nb,ydim = f_mean.shape
        

    #     f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
    #     #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            


    #     #if datatype == 'x':
    #     if datatype in ['x','xo']:            
    #         eps_samples = torch.randn(nb, ydim, nfsample).to(f_mean.device)                                
    #         eps_samples = eps_samples*f_ustd[:,:,None]                        #(nb,nout,nfsample)        

    #         if (self.static_mh.usemhbias) and ( indist_score[0].requires_grad):
    #             z_samples = torch.cat( [ torch.bernoulli(indist_score).unsqueeze(dim=-1) for _ in range(nfsample)],dim=-1)        
    #             eps_z_samples = eps_samples*(1 + z_samples) 
    #         else:
    #             eps_z_samples = eps_samples
    #         #eps_z_samples = eps_samples            
    #         f_std_samples = torch.einsum('bij,bjk->bik',f_u,eps_z_samples)
    #         #breakpoint()
            
    #     if datatype == 'xc':
    #         eps_samples = torch.randn(nb, ydim, nfsample).to(f_mean.device)                                
    #         eps_samples = eps_samples*f_ustd[:,:,None]                        #(nb,nout,nfsample)        
    #         f_std_samples = torch.einsum('bij,bjk->bik',f_u,eps_samples)            
    #         #breakpoint()

    #     f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                    
    #     f_samples = f_samples.permute(2,0,1)     
                   
    #     return f_samples
    
    
    #def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x'):        
    def compute_func_samples_conti_z2(self, f_mean, f_u, f_ustd ,indist_score, nfsample=5, datatype='x', upper_bound = 20):        
        
        #f_mean,f_Lcov = self.compute_dist_fs(x)        
        nb,ydim = f_mean.shape
        

        f_mean_samples = f_mean[:,:,None]  #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample)                           ##v61,v62
        #f_mean_samples = f_mean[:,:,None]*(1 - indist_score_samples) #(nb,nout,1)*(nb,1,nfsample) = (nb,nout,nfsample) ##v63 --> incurs spike            

        #self.fpriorstd


        #if datatype == 'x':
        if datatype in ['x','xt','xc','xo']:            
            eps_samples = torch.randn(nb, ydim, nfsample).to(f_mean.device)                               
            eps_samples = eps_samples*f_ustd[:,:,None]                        #(nb,nout,nfsample)        
            f_vstd_samples = torch.einsum('bij,bjk->bik',f_u,eps_samples)
            
            #f_priorstd = indist_score*self.fpriorstd
            f_priorstd_samples = self.f_priorstd*torch.randn(nb, ydim, nfsample).to(f_mean.device)
            #breakpoint()
            
            #if indist_score.sum().requires_grad:
            #    breakpoint()  
            f_std_samples = (1-indist_score[...,None])*f_vstd_samples + indist_score[...,None]*f_priorstd_samples
            

        f_samples = f_mean_samples + f_std_samples #(nb,nout,nfsample)                    
        f_samples = f_samples.permute(2,0,1)                        
        #return f_samples        
        return torch.clamp(f_samples,-upper_bound,upper_bound)


    # ##for fast training
    # ## predictive entrophy + new regularization term
    # def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
    #     if configs is not None:
    #         nctset = configs.train.nctset 
    #         reglambda = configs.train.reglambda
    #         nfsample = configs.train.nfsample
    #     else:            
    #         nctset = 2 
    #         reglambda = 1e-3
    #         nfsample = 1        
                
    #     #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
    #     self.train()            
    #     # get params
    #     #param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     q_outs_x          = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
    #     logits_sample_x   = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        
        
    #     ## set variational distribution on xc
    #     # q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
    #     q_outs_xc          = self.compute_func_dist2( xc, param_q_mean, param_q_logstd, datatype='xc')     
    #     logits_sample_xc   = self.compute_func_samples_conti_z2(q_outs_xc.f_mean, q_outs_xc.f_u, q_outs_xc.f_ustd, q_outs_xc.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        

    #     ent_x = self.get_entropy(logits_sample_x)[1]
    #     rent_x =  ent_x.max(dim=0)[0].reshape(-1,1)
                
    #     ent_xc = self.get_entropy(logits_sample_xc)[1]
    #     rent_xc = ent_xc.max(dim=0)[0].reshape(-1,1)
                    
                    
        
    #     # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
    #     self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
        
        
    #     #breakpoint()
    #     q_logits    = (logits_sample_x        , logits_sample_xc)
    #     q_std       = (q_outs_x.f_ustd        , q_outs_xc.f_ustd)                   
    #     q_indistscore = (q_outs_x.indist_score, q_outs_xc.indist_score)             
    #     q_mhdistance = (q_outs_x.mh_distance  , q_outs_xc.mh_distance)            
    #     p_indistscore = (rent_x               , rent_xc)

        
    #     #breakpoint()

                           
    #     # outputs        
    #     outs = AttrDict()
    #     outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
    #     #outs.logits_xc = logits_sample_xc 
    #     #outs.logits    = torch.log(q_pred_sample_x  + eps_log)                #logits_sample_x,logits_sample_xc
    #     #outs.logits_xc = torch.log(p_pred_sample_xc + eps_log)
        
    #     #outs.log_qf,outs.log_pf = log_qf, log_pf
    #     #outs.log_qf,outs.log_pf = None,None
    #     #outs.sup_kl = sup_kl                
    #     #outs.reg   = sup_kl           
    #     outs.qfstd = q_std   
    #     outs.mhdistance = q_mhdistance   
    #     outs.indistscore = q_indistscore        
    #     outs.pseudolabel = p_indistscore         
    #     return outs                     
    





    ##for fast training
    ## predictive entrophy + new regularization term
    def predict_train(self, x , y , xc , current_epoch=1,  trainreg=False , p_params=None,configs=None):                
        
        if configs is not None:
            nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            # nfsample = configs.train.nfsample
            nfsample = 1
            
        else:            
            nctset = 2 
            reglambda = 1e-3
            nfsample = 1        
                
        #print('ncset: {}, reglambda: {}, nfsample: {}'.format(nctset,reglambda,nfsample))        
        self.train()            
        # get params
        #param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
        q_outs_x          = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        logits_sample_x   = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        
        
        # ## set variational distribution on xc
        # # q_outs_xc          = self.compute_func_dist( xc, param_q_mean, param_q_logstd, datatype='xc')     
        # q_outs_xc          = self.compute_func_dist2( xc, param_q_mean, param_q_logstd, datatype='xc')     
        # logits_sample_xc   = self.compute_func_samples_conti_z2(q_outs_xc.f_mean, q_outs_xc.f_u, q_outs_xc.f_ustd, q_outs_xc.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)
        

        ent_x = self.get_entropy(logits_sample_x)[1]
        rent_x =  ent_x.max(dim=0)[0].reshape(-1,1)
                
        # ent_xc = self.get_entropy(logits_sample_xc)[1]
        # rent_xc = ent_xc.max(dim=0)[0].reshape(-1,1)
                    
                    
        
        # ## update mh mean.sum() and cov.sum() every batch, which will be used per epoch 
        self.static_mh.update_layers_static_batch(q_outs_x.h_list, y, datatype = 'x')                  
        
        
        #breakpoint()
        # q_logits    = (logits_sample_x        , logits_sample_xc)
        # q_std       = (q_outs_x.f_ustd        , q_outs_xc.f_ustd)                   
        # q_indistscore = (q_outs_x.indist_score, q_outs_xc.indist_score)             
        # q_mhdistance = (q_outs_x.mh_distance  , q_outs_xc.mh_distance)            
        # p_indistscore = (rent_x               , rent_xc)

        q_logits    = (logits_sample_x        , 0)
        q_std       = (q_outs_x.f_ustd        , 0)                   
        q_indistscore = (q_outs_x.indist_score, 0)             
        q_mhdistance = (q_outs_x.mh_distance  , 0)            
        p_indistscore = (rent_x               , 0)
        
        #breakpoint()

                           
        # outputs        
        outs = AttrDict()
        outs.logits    = logits_sample_x                #logits_sample_x,logits_sample_xc
        #outs.logits_xc = logits_sample_xc 
        #outs.logits    = torch.log(q_pred_sample_x  + eps_log)                #logits_sample_x,logits_sample_xc
        #outs.logits_xc = torch.log(p_pred_sample_xc + eps_log)
        
        #outs.log_qf,outs.log_pf = log_qf, log_pf
        #outs.log_qf,outs.log_pf = None,None
        #outs.sup_kl = sup_kl                
        #outs.reg   = sup_kl           
        outs.qfstd = q_std   
        outs.mhdistance = q_mhdistance   
        outs.indistscore = q_indistscore        
        outs.pseudolabel = p_indistscore         
        return outs                     
    


    def get_entropy(self, logit_samples,  scale=2.3):    
        
        pred_samples     = F.softmax(logit_samples, dim=-1).detach().clone()  #(nfsample, nb, nout)

        log_pred_samples = torch.log(pred_samples + eps_log)      #(nfsample, nb, nout)        
        entropy_samples  = -torch.sum(pred_samples*log_pred_samples,dim=-1)  #(nfsample, nb)
        rescaled_entropy_samples = torch.clamp(entropy_samples /scale , min=0.0, max=1.0)  #(nfsample, nb)
        return entropy_samples, rescaled_entropy_samples









    # def compute_sup_kl(self, params_qf,params_pf, Lowrank=True, Diag=True, nctset=2, eps=1e-4):
        
    #     def compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
    #         qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=True)
    #         pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=True)
    #         #breakpoint()
    #         return kl_divergence(qf_dist,pf_dist).sum(dim=-1,keepdim=True).mean(dim=0)
    
        
    #     # set params
    #     qf_mean,qf_Lcov = params_qf
    #     pf_mean,pf_Lcov = params_pf
    #     #zitter_mat = eps*torch.ones_like(qf_mean).to(qf_mean.device)        

    #     # set ctxset
    #     num_batch = qf_mean.shape[0] // nctset
    #     idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

    #     # compute kl
    #     kl_divergence_list = []
    #     for i_set in idx_set:
    #         i_params_q = qf_mean[i_set] , qf_Lcov[i_set]
    #         i_params_p = pf_mean[i_set] , pf_Lcov[i_set]       

    #         i_kl = compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=True )                    
    #         kl_divergence_list.append(i_kl)
 
    #     #breakpoint()
    #     kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
    #     return torch.clamp(kl_divergence_list.max(),min=eps_clamp)    



    
        #entropy = np.sum(-(logits + eps_log ) * np.log(preds + eps_log ), axis=1)





    # def predict_eval(self, x , nfsample=1):        
    #     #print('nfsample: {} in eval'.format(nfsample))        
    #     self.eval()        
    #     # set params
    #     param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
                
    #     # compute logit samples
    #     q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        
    #     #logits_sample = self.compute_func_samples(q_outs_x.f_mean,outs_x.f_Lcov, nfsample=nfsample) #(nfsample,nb,nout)
    #     # logits_sample    = self.compute_func_samples_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
    #     logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)


    #     rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
    #     beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


    #     pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)               
    #     outs.logits = logits_sample
    #     outs.stdfs = q_outs_x.stdonfs
    #     outs.indist_score = q_outs_x.indist_score
    #     outs.mh_distance = q_outs_x.mh_distance 
    #     outs.pseudo_label = beta_samples
        
    #     return outs                 
                                                           
        
        
        


    def predict_eval(self, x , nfsample=1):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')     
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, q_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        # rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        # beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        # p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_mean, p_outs_x.f_u, p_outs_x.f_ustd, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        # rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        # beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        outs.fustd = q_outs_x.f_ustd
        outs.indist = q_outs_x.indist_score
        outs.mhd = q_outs_x.mh_distance 

        outs.fstd = q_outs_x.f_std
        # outs.indist_score = q_outs_x.indist_score
        # outs.mh_distance = q_outs_x.mh_distance 
        # outs.pseudo_label = beta_samples
        # outs.pseudo_labelp = beta_samples_p        
        return outs                 
                                                           


    def predict_eval_oracle(self, x , nfsample=1 ,datatype = 'x'):        
        #print('nfsample: {} in eval'.format(nfsample))        
        self.eval()        
        # set params
        param_q_mean, param_q_logstd  = list(self.stochastic_nn.parameters()) ,  list(self.params_q_logstd)     
        param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        
                
        # compute logit samples
        # q_outs_x = self.compute_func_dist( x, param_q_mean, param_q_logstd, datatype='x')           
        # logits_sample    = self.compute_func_samples_conti_z(q_outs_x.f_mean, q_outs_x.f_Lcov, q_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        q_outs_x           = self.compute_func_dist2( x, param_q_mean, param_q_logstd, datatype='x')

        if datatype == 'x':
            oracle_indist_score = torch.zeros_like(q_outs_x.indist_score).to(x.device)
        if datatype == 'xo':
            oracle_indist_score = torch.ones_like(q_outs_x.indist_score).to(x.device)             
        logits_sample    = self.compute_func_samples_conti_z2(q_outs_x.f_mean, q_outs_x.f_u, q_outs_x.f_ustd, oracle_indist_score, nfsample=nfsample, datatype=datatype) #(nfsample,nb,nout)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 

        # rescaled_entropy_samples = self.get_entropy(logits_sample)[1]
        # beta_samples =  rescaled_entropy_samples.max(dim=0)[0].reshape(-1,1)


        # param_p_mean, param_p_logstd  = list(self.params_p_mean)              ,  list(self.params_p_logstd)                   
        # # p_outs_x = self.compute_func_dist( x, param_p_mean, param_p_logstd, datatype='x')           
        # # logits_sample_p    = self.compute_func_samples_conti_z(p_outs_x.f_mean, p_outs_x.f_Lcov, p_outs_x.indist_score, nfsample=nfsample) #(nfsample,nb,nout)
        # p_outs_x = self.compute_func_dist2( x, param_p_mean, param_p_logstd, datatype='x')           
        # logits_sample_p    = self.compute_func_samples_conti_z2(p_outs_x.f_ mean, p_outs_x.f_u, p_outs_x.f_std, p_outs_x.indist_score, nfsample=nfsample, datatype='x') #(nfsample,nb,nout)


        # rescaled_entropy_samples_p = self.get_entropy(logits_sample_p)[1]        
        # beta_samples_p =  rescaled_entropy_samples_p.max(dim=0)[0].reshape(-1,1)


                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)               
        outs.logits = logits_sample
        #outs.stdfs = q_outs_x.stdonfs
        # outs.indist_score = q_outs_x.indist_score
        # outs.mh_distance = q_outs_x.mh_distance 
        # outs.pseudo_label = beta_samples
        # outs.pseudo_labelp = beta_samples_p        
        return outs                         