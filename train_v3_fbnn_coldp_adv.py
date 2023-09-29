import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from collections.abc import Mapping
from torchvision import datasets, transforms
from attrdict import AttrDict

import os
import argparse
import time
from time import gmtime, strftime


from datasets_v2 import setup_dataset
from evaluate_v2 import get_acc,get_ll

#from models_shallownet import *
from models_shallownet import *
#from models_deepnet import *
#from models_deepnet_v2 import *
from models_deepnet_v3 import *


from train_v3_base import (setup_randomness,
                        setup_savedir,
                        update_configs_by_args,
                        setup_model,
                        train_epochs,
                        eval_trainedmodel,                        
)


# def setup_randomness(random_seed= 1111):
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(random_seed)
#     random.seed(random_seed)
#     return 


#from torch.nn
from functorch import make_functional_with_buffers



#omit_keys = ['nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint']
omit_keys = ['nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','ntrainset','nwarmupepoch',
             'nepoch','nbatch',
             'mhquantile','mhnepochprior','targetacc','nmhlayer',
             'userelreg','regcls','betahyp','updateperiod',
             'priorstdin','usepretrain','useregweight',
             'temppeak','topk','Tmax','lr',
             'mixcontext','mixself',
             ]

#'mixcontext',

#'priorstdin','priorstdout','usepretrain','useregweight','topk',

# 'usepretrain',
#,'nepochratio',


def setup_savedir(configs:AttrDict, gentime:str, omit_keys =omit_keys ):
    
    def setup_savepath(configs, omit_keys =omit_keys):
        #omit_keys = ['nperiod']
        full_name = ''
        for ith_key in configs:
            tmp_name = ''
            for jth_key in configs[ith_key]:
                if jth_key not in omit_keys:
                    tmp_name+='{}{}_'.format(jth_key,configs[ith_key][jth_key])
                else:
                    pass
            #print(tmp_name)
            full_name += tmp_name        
        return full_name[:-1]
    
    #savedir = './results/{}/{}/'.format(configs.dataset.trainset,setup_savepath(configs))
    savedir = './results/{}_{}/{}/'.format(configs.dataset.trainset,configs.dataset.contextset,setup_savepath(configs))
    
    1 if os.path.isdir(savedir) else os.makedirs(savedir)
    #savepath = savedir + 'seeds{}_gentime{}'.format(str(args.seeds),str(args.gentime)) + '.pth'
    #savepath = savedir + 'seeds{}_gentime{}'.format(str(args.seeds),str(args.gentime)) + '.pth'        
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth'     
    
    print('savepath: {}'.format(savepath))           
    return savepath



#def loss_fn(outs,targets,configs , model):
def loss_fn(outs,targets,configs,use_bayes=False, beta_coldposterior=1e-8 ):
    #loss_negll = nn.CrossEntropyLoss()(outs.logits,targets)          
       
    def negloss_elbo(outs,targets):
        #set loss
        loss_fun = nn.CrossEntropyLoss(reduction='none')            
        #compute loss        
        num_sample,nb, nout = outs.logits.shape
        y_samples = targets.view(1,-1).repeat(num_sample,1) #(num_sample,nb)
        neglog_y_given_f = loss_fun(outs.logits.reshape(num_sample*nb,-1), y_samples.reshape(-1))
        
        #breakpoint()
        return neglog_y_given_f.mean()

    def negloss_iwae(outs,targets):
        #set loss
        loss_fun = nn.CrossEntropyLoss(reduction='none')            
        #compute loss        
        num_sample,nb,nout = outs.logits.shape
        y_samples = targets.view(1,-1).repeat(num_sample,1) #(num_sample,nb)
        neglog_y_given_f = loss_fun(outs.logits.reshape(num_sample*nb,-1), y_samples.reshape(-1))
        log_y_given_f = -neglog_y_given_f.reshape(num_sample,nb)                             ## over samples
        log_y_given_x = (torch.logsumexp(log_y_given_f,dim=0) - np.log(num_sample)).mean()   ## over batches     
        
        #breakpoint()        
        return -log_y_given_x
        
    #loss_fn = negloss_elbo if use_bayes else negloss_iwae            
    loss_fn = negloss_elbo if configs.train.losstype == 'elbo' else negloss_iwae            
    loss_negll = loss_fn(outs,targets)            
    #loss_reg = float(configs.train.reglambda)*outs.reg    
    
    #breakpoint()
    #loss_reg = float(configs.train.reglambda)*outs.reg  if use_bayes else  beta_coldposterior*outs.reg
    #loss_reg = float(configs.train.reglambda)*outs.reg  if use_bayes else  torch.zeros(1).to(loss_negll.device)
    loss_reg = float(configs.train.reglambda)*outs.reg 
    
    loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl

    return loss, loss_negll, loss_reg





# FGSM attack code
def fgsm_attack(image, data_grad ,  epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range

    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    min,max = image.min(),image.max()
    perturbed_image = torch.clamp(perturbed_image, min, max)

    return perturbed_image



import torch.nn.functional as F

def  train_epochs_with_prior(trainloader,contextloader, model, optimizer, loss_fn, scheduler, configs=None, use_bayes=False, updated_p_params =None):
        
    loss_list,lossdata_list,lossreg_list = [],[],[]
    for i, ((x, y),(xc, yc)) in enumerate(zip(trainloader,contextloader) ):    
        
        # compute loss         
        logit_samples = model.compute_logit_samples(x.cuda(), use_bayes=use_bayes, configs=configs)                                 


        if use_bayes and (float(configs.train.reglambda) > 0.0) :
            # compute sup kl    
            #breakpoint()            
    
            # if configs.train.mixcontext:
                                             
            #     alpha = configs.train.mixalpha*torch.rand(xc.shape[0]).view(-1,1,1,1)                
            #     if configs.train.mixself:
            #         xc = alpha*x[torch.randperm(x.shape[0])] + (1-alpha)*x
            #     else:
            #         xc = alpha*xc + (1-alpha)*x                


            ## build adversial examples as contextset
            x_copy = x.clone().detach().requires_grad_(True)
            logit_copy_samples = model.compute_logit_samples(x_copy.cuda(), use_bayes=False, configs=configs)                                 
            loss_tmp = F.cross_entropy( logit_copy_samples.view(np.prod(logit_copy_samples.shape[:2]),-1)  , y.cuda().view(-1)  )
            loss_tmp.backward()            
            data_grad = x_copy.grad.data
            xc = fgsm_attack(x, data_grad, epsilon= configs.train.mixalpha)
            model.zero_grad()


            params_qf, params_pf, weight = model.compute_fdist(xc.cuda(), p_params = updated_p_params, configs=configs)            
            sup_kl = model.compute_sup_kl(params_qf, params_pf, weight=weight, Lowrank=False, Diag=True, nctset=configs.train.nctset )   

            #breakpoint() #
        else:
            sup_kl = torch.zeros(1).to(logit_samples.device)


        
        
        #breakpoint()
            
        outs = AttrDict()
        outs.logits = logit_samples
        outs.reg = sup_kl                               
        loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs ,use_bayes=use_bayes)


        # update param
        optimizer.zero_grad()
        loss.backward()
        #breakpoint()
        
        optimizer.step()    
        scheduler.step()
        

        # record metrics
        loss_list.append(loss.item())        
        lossdata_list.append(loss_data.item())        
        lossreg_list.append(loss_reg.item())
        
        # empty cuda
        torch.cuda.empty_cache()        
        
        
    outs = AttrDict()
    outs.avg_loss = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg = np.array(lossreg_list).mean().round(3)    
    return outs




def eval_trainedmodel(dataloader, model,q_params = None , nfsample=1):
    assert q_params is not None
    pred_list,label_list = [],[]    
    logit_diversity_list = []

    model.eval()    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):    
            # predict class            
            outs = model.predict_eval(x.cuda(), nfsample=nfsample)                     

            # compute performance
            pred_list.append(outs.pred.cpu())
            label_list.append(y.cpu())
            
            torch.cuda.empty_cache() 
    
    pred_list = torch.cat(pred_list,dim=0)    
    label_list = torch.cat(label_list,dim=0)    
     
    #breakpoint()
    # evaluate model
    outs = AttrDict()
    outs.pred_list = pred_list
    outs.label_list = label_list    
    #outs.logit_diversity_list = logit_diversity_list
    return outs    




# def train_mixutre_module(trainloader, model, configs=None):
#     # for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):            
#     for i, (x, y) in enumerate(trainloader ):            
#         #update mixture feature         
#         model.update_hidden_module(x.cuda(), y.cuda(), configs=configs)                                 
#     return 



eps_log=1e-8
def process_prior_parameters_list(prior_parameters_list,configs=None):
    assert len(prior_parameters_list) >= 2
    nprior_model = len(prior_parameters_list)

    #priorscale = configs.model.priorstdout if configs is not None else 1e-3
    priorscale = 1e0 if configs is not None else 1e-3

    # process mean 
    updated_p_mean = prior_parameters_list[0]
    for current_path_list in prior_parameters_list[1:]:
        for ith_updated, ith_current in zip(updated_p_mean, current_path_list):
            ith_updated.data += (1/nprior_model)*ith_current.data

    # process logstd
    updated_p_logstd = [torch.log(priorscale*torch.ones_like(i_p_mean) + eps_log).to(i_p_mean) for i_p_mean in updated_p_mean]
    
    updated_p_params = (updated_p_mean,updated_p_logstd)
    return updated_p_params






#tmp_logger_path = './training_log_fbnnsp2.txt'
#tmp_logger_path = './training_log_fbnnsp3.txt'
#tmp_logger_path = './training_log_fbnnsp4.txt'
#tmp_logger_path = './training_log_fbnnsp5.txt'
#tmp_logger_path = './training_log_fbnnsp6.txt'
#tmp_logger_path = './training_log_fbnnsp7_cold.txt'
tmp_logger_path = './training_log_v3_fbnnsp7_coldadv.txt'

def write_msg(current_msg:str,withprint=True,tmp_logger_path=tmp_logger_path):
    if withprint:
        print(current_msg)
    
    with open(tmp_logger_path, "a") as file:
        file.write(current_msg + '\n')
        file.close()    



def main(args):
    use_bayes=False
    p_mean_weight = None
                                
    # get and update configs
    configs = update_configs_by_args(args)
    print('update configs: {}'.format(configs))

    # set randomness
    setup_randomness(random_seed = configs.exp.seeds)                         
                    
                        
    # set epoch list for prior construction
    epochs_for_bayes = int(configs.train.nepoch*configs.train.nepochratio)
    #epochs_for_bayes = 2
        
    #past_periods, period = 20 , 4
    # past_periods, period = 10 , 3    
    #past_periods, period = 10 , 2    
    #past_periods, period = 5 , 1    
    
    # tmp_list = np.arange( epochs_for_bayes  - past_periods, epochs_for_bayes ) 
    # epochs_for_priorconstruction = [i_tmp for i_tmp in tmp_list if i_tmp % period ==0]

    epochs_for_priorconstruction = []

    #epochs_for_bayes, epochs_for_priorconstruction = 3, [1,2]
    print('epochs for bayes: {}, \t epochs_for_priorconstruction: {}'.format(epochs_for_bayes, epochs_for_priorconstruction))

    # setup dataset
    trainloader,valloader, _, contextloader, _   =  setup_dataset(configs)    
    
    # build model 
    model,optimizer,scheduler = setup_model(configs)
    
    # set savepath
    savepath = setup_savedir(configs,gentime=args.gentime,omit_keys=omit_keys)
    write_msg(current_msg='savepath: {}'.format(savepath))    
    
    # train model
    p_params = None
    opt_val_acc,opt_val_nll = 0.0,1e8
    metrics = []
    
    prior_mean_param_list = []
    updated_p_params = None
    for i in range(1,configs.train.nepoch+1):


        # #if (i > int(configs.train.nepoch*configs.train.nepochratio) - 10) and  (i <= int(configs.train.nepoch*configs.train.nepochratio) - 1) and (i%5==0):                                    
        # #if (i == int(configs.train.nepoch*configs.train.nepochratio) - 10)  or (i == int(configs.train.nepoch*configs.train.nepochratio) - 5) :                                                            
        # #if (i > int(configs.train.nepoch*configs.train.nepochratio) - 10) and  (i <= int(configs.train.nepoch*configs.train.nepochratio) - 1) and (i%5==0):                                    
        # if (i >= int(configs.train.nepoch*configs.train.nepochratio) - 10)  and (i <= int(configs.train.nepoch*configs.train.nepochratio) - 1) :              
        # # if  i <=6:                     
        #     if i % 3 == 0 :                                                         
        #         prior_mean_param_list.append(model.get_current_p_params())



        # if i == int(configs.train.nepoch*configs.train.nepochratio):                                    
        # #if i == 6:                                                
        #     train_mixutre_module(trainloader,
        #                          model, 
        #                          configs=configs,
        #                          )
            
        #     #breakpoint()                       
        #     updated_p_params = process_prior_parameters_list(prior_mean_param_list,configs=configs)
        #     use_bayes=True
            
            
        #if i == int(configs.train.nepoch*configs.train.nepochratio):                                    
        if i == epochs_for_bayes:
            use_bayes=True
            


        # train model            
        t = time.time()            
        outs_train = train_epochs_with_prior(trainloader,
                                             contextloader, 
                                             model, 
                                             optimizer,
                                             loss_fn,
                                             scheduler,
                                             configs = configs, 
                                             use_bayes=use_bayes,
                                             updated_p_params = updated_p_params,
                                             )
        elapsed = time.time() - t
        


        # evaluate model on training set
        outs_eval_val =  eval_trainedmodel(valloader, model , q_params = list(model.parameters()) , nfsample= configs.train.nfsample)
        acc_val       =  get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
        nll_val       = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
        
        
        
        

        nfreqprint = 5 if configs.train.usepretrain else configs.exp.nfreqprint        
        #nfreqprint = 1
        if (i % nfreqprint == 0) or (i==1):        
            msg = '[{}/{}], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f}, \t usebayes: {} '.format(i,configs.train.nepoch,elapsed,
                                                                                                                                   outs_train.avg_loss,
                                                                                                                                   outs_train.avg_lossdata,
                                                                                                                                   outs_train.avg_lossreg,
                                                                                                                                   acc_val,
                                                                                                                                   nll_val,
                                                                                                                                   use_bayes)

            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val))
            
            
            # if val_acc < outs_eval_val.acc:
            #     val_acc = outs_eval_val.acc            
            # if opt_val_nll > nll_val:
            if (opt_val_nll > nll_val) and use_bayes:
            
                opt_val_nll = nll_val                 
                save_dict = {'args':args,
                             'configs':configs,
                             'metrics':metrics,
                             'state_dict': model.state_dict(),
                             'optimizer_dict': model.state_dict(),
                             'acc_val':acc_val,
                             'nll_val':nll_val,                             
                             'epochs':i,
                }
                torch.save(save_dict, savepath)
                msg += '\t saved'
                
            ## write msg                    
            write_msg(current_msg='{}'.format(msg))                    
    
    torch.cuda.empty_cache()
    print('\n'*3)
    return 
    




if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='exp_fbnn')    
    # # default hyperparameters 
    # parser.add_argument('--seeds',type=int, default=1111)    
    # parser.add_argument('--nexp',type=int, default=1)   
    # parser.add_argument('--nfreqprint',type=int, default=5)   
    # parser.add_argument('--lr',type=float, default=.1)        
    # parser.add_argument('--nepoch',type=int, default=200 , help='number of epochs')            
        
    # # varying hyperparameters
    # parser.add_argument('--configs',type=str, default='abc')    
    # parser.add_argument('--modelname',type=str, default='resnet20')    
    # parser.add_argument('--reglambda',type=float, default=.1)    
    # parser.add_argument('--scalelogstd',type=int, default=-8)  
    # parser.add_argument('--nctset',type=int, default=0)        
    # parser.add_argument('--nfsample',type=int, default=1, help='number of samples for forward pass')  
    # parser.add_argument('--nsplitlayer',type=int, default=0, help='number of lastlayer')  

    parser.add_argument('--nexp',type=int, default=1)       
    parser.add_argument('--configs',type=str, default='abc')    
    parser.add_argument('--modelname',type=str, default='fbresnet20')    
    parser.add_argument('--reglambda',type=float, default=.1)    
    parser.add_argument('--scalelogstd',type=int, default=-8)      
    parser.add_argument('--nctset',type=int, default=1)        
    parser.add_argument('--nfsample',type=int, default=1, help='number of samples for forward pass')  
    
    #parser.add_argument('--userelreg', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ##add recently 05/03
    parser.add_argument('--priorstdout',type=float, default=10.)
    parser.add_argument('--nepochratio',type=float, default=.5)
    parser.add_argument('--v',type=int, default='1')    
    #parser.add_argument('--useregweight', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ##add recently 05/03
    parser.add_argument('--topk',type=int, default=10)        
    parser.add_argument('--losstype',type=str, default='elbo')    
    
    parser.add_argument('--mixcontext', action=argparse.BooleanOptionalAction, default=False, help='mix context set')
    parser.add_argument('--mixalpha', type=float, default=.5)
    parser.add_argument('--mixself', action=argparse.BooleanOptionalAction, default=False, help='mix true')

    # mixalpha: 0.5
    # mixself: False


     
    parser.add_argument('--seeds',type=int, default=1111)        
        
    args = parser.parse_args()
    args.gentime = strftime("%m%d%H%M",gmtime()) 

    #print(args.nepoch)
    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        

