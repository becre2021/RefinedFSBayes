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


from evaluate_v2 import get_acc,get_ll
from datasets_resnet import setup_dataset, _setup_mainset, _setup_oodset
from ood_results import entropy_auroc, save_seed_results, format_evaluation_metrics


from models_bayeslayers import Conv2d_VI
from models_deepnet import *


from train_map import (setup_randomness,
                        setup_savedir,
                        update_configs_by_args,
                        get_results_from_outs,            
                        get_results_from_outs_without_calibration,        
)


import copy


omit_keys = ['prior_distribution', 'oodset',
             'nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','nwarmupepoch',
             'nepoch','nbatch','Tmax',
             'mhquantile','mhnepochprior','targetacc','nmhlayer',
             'userelreg','regcls','betahyp','updateperiod','losstype',
             'priorstdin','useregweight',
             'mixcontext','mixself',     
             'swagrank','swagper',
              'temppeak',       
             ]


def on_random_sample(m):         
    if  isinstance(m, Conv2d_VI) :
        m.random_sample=True
    return

def off_random_sample(m):         
    #print('current_m: {}'.format(m))   
    if isinstance(m, Conv2d_VI) :
        m.random_sample=False
    return


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
    
    savedir = './results/{}_{}/{}/'.format(configs.dataset.trainset,configs.dataset.contextset,setup_savepath(configs))
    
    if 'prior_distribution' in configs.train:
        savedir = savedir.replace('./results/', './results/ablation_' + configs.train.prior_distribution + '/', 1)

    # 1 if os.path.isdir(savedir) else os.makedirs(savedir)    
    
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    #savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth'         
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth.tar'         

    print('savepath: {}'.format(savepath))           
    return savepath




def loss_fn(outs,targets,configs,use_bayes=False, beta_coldposterior=1e-6 ):
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
    loss_negll =  negloss_elbo(outs,targets)            
    

    loss_reg = float(configs.train.reglambda)*outs.reg  if use_bayes else  0.0*outs.reg 


    loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl
    return loss, loss_negll, loss_reg




def fgsm_attack(image, data_grad ,  epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image

    if epsilon >= 0.0 :
        perturbed_image = image + epsilon*sign_data_grad


    if epsilon == -1:        
        #epsilon_random = np.random.choice( [0.01, 0.05] ).item()        
        epsilon_random = 0.05 + 0.05*np.random.rand()
        perturbed_image = image + epsilon_random*sign_data_grad


    if epsilon == -2:        
        #epsilon_random = np.random.choice( [0.01, 0.05] ).item()        
        epsilon_random = 0.05 + 0.10*np.random.rand()
        perturbed_image = image + epsilon_random*sign_data_grad


    if epsilon == -3:        
        #epsilon_random = np.random.choice( [0.01, 0.05] ).item()        
        epsilon_random = 0.01 + 0.09*np.random.rand()
        perturbed_image = image + epsilon_random*sign_data_grad



    if epsilon == -4:        
        #epsilon_random = np.random.choice( [0.01, 0.05] ).item()        
        epsilon_random = 0.10 + 0.10*np.random.rand()
        perturbed_image = image + epsilon_random*sign_data_grad

    
    min,max = image.min(),image.max()
    perturbed_image = torch.clamp(perturbed_image, min, max)

    return perturbed_image




def  train_epochs_with_prior(trainloader,contextloader, model, optimizer, loss_fn, scheduler, configs=None, use_bayes=False, update_h=False):
    
    loss_list,lossdata_list,lossreg_list,memory_list = [],[],[],[]
    #for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):    
    for i, (x, y) in enumerate(trainloader):    
        
        # compute loss         
        #breakpoint()
        logit_samples, h_x = model.compute_logit_samples(x.cuda(), y.cuda(),  use_bayes=use_bayes, update_h = update_h, configs=configs)                                 


        # compute sup kl
        #if use_bayes:    
        
        if use_bayes and (float(configs.train.reglambda) > 0.0) :                   


            ## build adversial examples as h_x_contextset            
            # v10 - reported             
            # h_x_copy, h_x_grad           = model.compute_h_x_grad(h_x)
            # h_xc                         = fgsm_attack(h_x_copy, h_x_grad, epsilon= configs.train.mixalpha).detach().clone()
            # params_qf, params_pf, weight = model.compute_fdist_from_h(h_xc,  useswagparam=True) 

            
            h_x_copy, h_x_grad = model.compute_h_x_grad(h_x)            
            #h_xc               = h_x  +  float(configs.train.mixalpha)*h_x_grad.sign()                      
            h_xc                = h_x_copy +  float(configs.train.mixalpha)*h_x_grad.sign()       
            h_xc                = torch.clamp(h_xc, h_xc.min() , h_xc.max())               
            params_qf, params_pf, weight = model.compute_fdist_from_h(
                h_xc, useswagparam=configs.train.get('prior_distribution', 'trajectory') == 'trajectory')



            
            

            sup_kl = model.compute_sup_kl(params_qf, params_pf, weight=weight, Lowrank=False, Diag=True, nctset=configs.train.nctset )   
            memory_gpu     = torch.cuda.memory_reserved() / 1024 / 1024


            #breakpoint()
 
        else:
            sup_kl = torch.zeros(1).to(logit_samples.device)           
            memory_gpu     = -1

        
            
        outs = AttrDict()
        outs.logits = logit_samples
        outs.reg = sup_kl                               
        loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs ,use_bayes=use_bayes)

        #breakpoint()
        # update param
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        scheduler.step()
        

        # record metrics
        loss_list.append(loss.item())         
        lossdata_list.append(loss_data.item())        
        lossreg_list.append(loss_reg.item())
        memory_list.append(memory_gpu)
        # empty cuda
        torch.cuda.empty_cache()        

        
    outs = AttrDict()
    outs.avg_loss = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg = np.array(lossreg_list).mean().round(3)  
    outs.avg_memory = np.array(memory_gpu).mean().round(3)  

    #lr trace    
    outs.last_lr = np.array(scheduler.get_last_lr()[0]).round(3)  
    return outs




def eval_trainedmodel(dataloader, model,q_params = None , use_bayes=False, nfsample=1):
    assert q_params is not None
    pred_list,label_list = [],[]    
    logit_list = []

    model.eval()    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):    
            # predict class            
            outs = model.predict_eval(x.cuda(),use_bayes=use_bayes, nfsample=nfsample)                     
                                             
            # compute performance
            pred_list.append(outs.pred.cpu())
            label_list.append(y.cpu())
            logit_list.append(outs.logits.cpu())      


            torch.cuda.empty_cache() 
    
    pred_list = torch.cat(pred_list,dim=0)    
    label_list = torch.cat(label_list,dim=0)    
    logit_list = torch.cat(logit_list,dim=1)    
    
    #print('logit_list.shape: {} in eval_trainedmodel'.format(logit_list.shape))
        
    # evaluate model
    outs = AttrDict()
    outs.pred_list = pred_list
    outs.label_list = label_list    
    outs.logit_list = logit_list #(ns,nb,npred)
    return outs    





def train_mixutre_module(trainloader, model, configs=None):
    # for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):            
    for i, (x, y) in enumerate(trainloader ):            
        #update mixture feature         
        model.update_hidden_module(x.cuda(), y.cuda())

    return 




def process_priorparamlist_as_swagdiag(prior_parameters_list,configs=None, eps_log=1e-8, eps_clamp=1e-20):
    assert len(prior_parameters_list) >= 1
    nprior_model = len(prior_parameters_list)

    emp_first_moment   = [torch.zeros_like(ith_param).to(ith_param) for ith_param in prior_parameters_list[0]]
    emp_second_moment  = [torch.zeros_like(ith_param).to(ith_param) for ith_param in prior_parameters_list[0]]    
    
    for current_path_list in prior_parameters_list:
        for ith_first_moment, ith_second_moment, ith_current in zip(emp_first_moment,emp_second_moment,current_path_list):
            ith_first_moment.data  += (1/nprior_model)*ith_current.data
            ith_second_moment.data += (1/nprior_model)*torch.pow(ith_current.data,2)

    # updated_p_logstd = [torch.log( torch.sqrt(ith_second_moment - ith_first_moment**2 + eps_log) ).to(ith_first_moment) for ith_first_moment,ith_second_moment in zip(emp_first_moment,emp_second_moment)]
    updated_p_logstd = [torch.log( torch.sqrt( torch.clamp(ith_second_moment - ith_first_moment**2, min=eps_clamp)) ).to(ith_first_moment) for ith_first_moment,ith_second_moment in zip(emp_first_moment,emp_second_moment)]
    updated_p_mean   = emp_first_moment    
    
    updated_p_params = (updated_p_mean,updated_p_logstd)
    return updated_p_params






def write_msg(current_msg:str,withprint=True,tmp_logger_path='./a.txt'):
    if withprint:
        print(current_msg)
    
    with open(tmp_logger_path, "a") as file:
        file.write(current_msg + '\n')
        file.close()    




import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def setup_scheduler_waramupcosdecay(optimizer, nwarmupstep:int = 5, ncosinestep:int = 10, lr_warmup: float = 1e-1 ):
    pi = math.pi
    milestones = [nwarmupstep]
    
    #print('configs.train.lr: {}, (step / nwarmupstep)): {} in scheduler'.format(lr_warmup, (step / nwarmupstep)) )
    
    ## set lr function
    linear_lr = lambda step:  0*(1 - (step / nwarmupstep)) + lr_warmup*(step / nwarmupstep)
    cosine_decay = lambda step:  lr_warmup * 0.5 * (1 + np.cos(pi * step/ ncosinestep ))
    
    scheduler1 = LambdaLR(optimizer, lr_lambda=linear_lr)  # optimizer.lr * linear_lr.get_lr()
    scheduler2 = LambdaLR(optimizer, lr_lambda=cosine_decay )
    scheduler = SequentialLR(optimizer, milestones=milestones, schedulers=[scheduler1, scheduler2])    
    return scheduler





def main(args):

                                
    # get and update configs
    eval_checkpoint = getattr(args, 'eval_checkpoint', None)
    if eval_checkpoint:
        checkpoint = torch.load(eval_checkpoint, map_location='cpu', weights_only=False)
        configs = checkpoint['configs']
        saved_args = checkpoint.get('args')
        for option in ('oodsets', 'places365_root', 'result_json', 'result_history', 'table_seeds', 'table_nfsamples', 'table_ddof', 'table_filename'):
            if not getattr(args, option, None):
                setattr(args, option, getattr(saved_args, option, None))
        del checkpoint
    else:
        configs = update_configs_by_args(args)
    if not eval_checkpoint:
        configs['train']['eval_train'] = getattr(args, 'eval_train', False)
        if getattr(args, 'prior_distribution', None) is not None:
            configs['train']['prior_distribution'] = args.prior_distribution
    oodsets = getattr(args, 'oodsets', None) or [configs.dataset.oodset]
    configs['dataset']['oodset'] = oodsets[0]
    if getattr(args, 'places365_root', None):
        os.environ['PLACES365_ROOT'] = args.places365_root
    print('update configs: {}'.format(configs))

    # set randomness
    setup_randomness(random_seed = configs.exp.seeds)                         
                    
    # setup dataset
    if eval_checkpoint:
        _, _, testloader = _setup_mainset(configs)
        oodloader = _setup_oodset(configs)
    else:
        trainloader,valloader, testloader, contextloader, oodloader = setup_dataset(configs)
    oodloaders = {oodsets[0]: oodloader}
    for ood_name in oodsets[1:]:
        ood_configs = copy.deepcopy(configs)
        ood_configs['dataset']['oodset'] = ood_name
        oodloaders[ood_name] = _setup_oodset(ood_configs)

    if eval_checkpoint:
        model = _setup_model_deep(configs).cuda()
        model.deterministic_nn.apply(on_random_sample)
        evaluate_saved_checkpoint(eval_checkpoint, args, model, testloader, oodloaders, configs)
        return

    
    
    # set savepath
    tmp_logger_path = './v{}_seed{}_trainlog_{}_{}new.txt'.format(configs.exp.v,configs.exp.seeds,configs.dataset.trainset,configs.model.modelname)  #retrain
    tmp_test_path   = './totalresults_{}_{}new.txt'.format(configs.dataset.trainset,configs.model.modelname)  #retrain



    savepath = setup_savedir(configs,gentime=args.gentime,omit_keys=omit_keys)
    write_msg(current_msg='savepath: {}'.format(savepath),tmp_logger_path=tmp_logger_path)    


    
    ## set epoch list for prior construction
    #nfreqprint       = configs.exp.nfreqprint            
    #epochs_for_bayes = int(configs.train.nepoch*configs.train.nepochratio)      
        
    # if configs.train.paramtrace == 9:
    #     past_periods, period = 10 , 2
    #     tmp_list = np.arange( epochs_for_bayes - past_periods, epochs_for_bayes) 
    #     epochs_for_priorconstruction = [i_tmp for i_tmp in tmp_list if i_tmp % period ==0]

    # if configs.train.paramtrace == 10: ## 1 simlar
    #     past_periods, period = 20 , 4
    #     tmp_list = np.arange( epochs_for_bayes - past_periods, epochs_for_bayes) 
    #     epochs_for_priorconstruction = [i_tmp for i_tmp in tmp_list if i_tmp % period ==0]

    # if configs.train.paramtrace == 11:
    #     past_periods, period = 30 , 6
    #     tmp_list = np.arange( epochs_for_bayes - past_periods, epochs_for_bayes ) 
    #     epochs_for_priorconstruction = [i_tmp for i_tmp in tmp_list if i_tmp % period ==0]    

        
    # if configs.train.paramtrace == -1:            
    #     epochs_for_bayes, epochs_for_priorconstruction = 3, [1,2]
    #     nfreqprint = 1


    # nfreqprint                   = configs.exp.nfreqprint            
    # epochs_for_bayes             = int(configs.train.nepoch*configs.train.nepochratio)    
    # past_periods                 = int(configs.train.nepoch*0.1) 
    # period                       = 2
    
    # tmp_list                     = np.arange( epochs_for_bayes - past_periods, epochs_for_bayes) 
    # epochs_for_priorconstruction = [i_tmp for i_tmp in tmp_list if i_tmp % period ==0]
    
    #breakpoint()
    nfreqprint                   = configs.exp.nfreqprint            
    epochs_for_bayes             = int(configs.train.nepoch*configs.train.nepochratio)    
    past_periods                 = int(configs.train.nepoch*0.1) 
    ntrajectory                  = 4
    epochs_for_priorconstruction = np.linspace( epochs_for_bayes - past_periods, epochs_for_bayes, ntrajectory,dtype=int) 

    

    print('epochs for bayes: {}, \t epochs_for_priorconstruction: {}'.format(epochs_for_bayes, epochs_for_priorconstruction))
    
    


    ## not use pretrain, i.e., scratch training
    # # set randomness
    # setup_randomness(random_seed = configs.exp.seeds)                         
    
    # build model 
    #model,optimizer,scheduler = setup_model(configs)

    
    print('get deep model: {}'.format(configs.model.modelname))      
    model = _setup_model_deep(configs)
        
    model = model.cuda() if torch.cuda.is_available() else model
    #breakpoint()
    #model.turnon_learn_var(learn_var=False)
    model.deterministic_nn.apply(off_random_sample)
    
    
    optimizer = torch.optim.SGD(model.parameters(), 
                                #lr= float(configs.train.lr) ,
                                lr= 1. ,                                
                                momentum=0.9,
                                weight_decay=5e-4,
                                )

    nstep = configs.dataset.ntrainset //configs.train.nbatch + 1
    nwarmupstep = configs.train.nwarmupepoch*nstep
    nlaststep   = configs.train.nepoch*nstep
    ncosinestep = nlaststep - nwarmupstep

    
    scheduler = setup_scheduler_waramupcosdecay(optimizer ,
                                                nwarmupstep = nwarmupstep, 
                                                ncosinestep = ncosinestep, 
                                                lr_warmup= float(configs.train.lr))    

    
    
    # set epochs
    start_epoch,end_epoch = 1, configs.train.nepoch+1

    # set params
    opt_val_acc,opt_val_nll = 0.0,1e8
    metrics = []    
    prior_param_list = []
    updated_p_params = None
    use_bayes=False    
    
    

    
    
    #for i in range(1,configs.train.nepoch+1):
    for i in range(start_epoch,end_epoch):


        # set prior params       
        #if i == epochs_for_bayes and (not configs.train.usepretrain):                                    
        if i == epochs_for_bayes :                                    
            #breakpoint()                       
            updated_p_params = process_priorparamlist_as_swagdiag(prior_param_list,configs=configs)
            model.update_params_p(*updated_p_params)
            model.deterministic_nn.apply(on_random_sample)            
            use_bayes=True


        # train model            
        t = time.time()            
        outs_train = train_epochs_with_prior(trainloader,
                                             contextloader, 
                                             model, 
                                             optimizer,
                                             loss_fn,
                                             scheduler,
                                             configs   = configs, 
                                             use_bayes = use_bayes,
                                             update_h  = True if i in epochs_for_priorconstruction else False,
                                             )
        elapsed = time.time() - t
        


        # set prior params collection
        if i in epochs_for_priorconstruction:
            # collect last layer params
            prior_param_list.append(model.get_current_p_params())
        
        

        # set evaluate trained models
        if ((i % nfreqprint == 0) or (i==int(configs.train.nepoch))  or  (i ==1)):
            train_metrics_msg = ''
            if getattr(args, 'eval_train', False):
                outs_eval_tr = eval_trainedmodel(
                    trainloader, model, q_params=list(model.parameters()),
                    use_bayes=use_bayes, nfsample=configs.train.nfsample)
                acc_tr = get_acc(outs_eval_tr.pred_list.cpu().data.numpy(),
                                 outs_eval_tr.label_list.cpu().data.numpy())
                nll_tr = -get_ll(outs_eval_tr.pred_list.cpu().data.numpy(),
                                 outs_eval_tr.label_list.cpu().data.numpy())
                train_metrics_msg = '\t acc_tr {:.3f}, \t nll_tr {:.3f}'.format(acc_tr, nll_tr)

            # evaluate model on validation set
            outs_eval_val = eval_trainedmodel(valloader,
                                              model ,
                                              q_params = list(model.parameters()) , 
                                              use_bayes=use_bayes,
                                              nfsample= configs.train.nfsample)
            
            
            #breakpoint()
            
            acc_val       =  get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            nll_val       = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            
            
            
 
            msg = '[{}/{}], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f}, \t last_lr {:.3f}, \t usebayes: {} '.format(i,configs.train.nepoch,elapsed,
                                                                                                                                                                                            outs_train.avg_loss,
                                                                                                                                                                                            outs_train.avg_lossdata,
                                                                                                                                                                                            outs_train.avg_lossreg,
                                                                                                                                                                                            acc_val,
                                                                                                                                                                                            nll_val,
                                                                                                                                                                                            outs_train.last_lr,
                                                                                                                                                                                            use_bayes)
            msg += train_metrics_msg
            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val, outs_train.avg_memory))
            
            
            # if val_acc < outs_eval_val.acc:
            #     val_acc = outs_eval_val.acc            
            
            #if opt_val_nll > nll_val:
            if (opt_val_nll > nll_val) and use_bayes:
            
                opt_val_nll = nll_val                 
                save_dict = {'args':args,
                             'configs':configs,
                             'metrics':metrics,
                            #  'state_dict': copy.deepcopy(model).state_dict(),
                            #  'optimizer_dict': copy.deepcopy(optimizer).state_dict(),
                            #  'scheduler_dict': copy.deepcopy(scheduler).state_dict(),
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer_dict': copy.deepcopy(optimizer.state_dict()),
                             'scheduler_dict': copy.deepcopy(scheduler.state_dict()),
                             'acc_val':acc_val,
                             'nll_val':nll_val,                             
                             'epochs':i,
                             #'prior_param_list': prior_param_list,
                             #'updated_p_params' : updated_p_params,
                }
                torch.save(save_dict, savepath)
                msg += '\t saved'
                
               
                
            ## write msg                    
            write_msg(current_msg='{}'.format(msg),tmp_logger_path=tmp_logger_path)                    
    
    torch.cuda.empty_cache()
    print('\n'*3)
    
    



    ## --------------------------------------------------
    evaluate_saved_checkpoint(savepath, args, model, testloader, oodloaders, configs, metrics)
    return


def evaluate_saved_checkpoint(savepath, args, model, testloader, oodloaders, configs, metrics=None):
    """Evaluate a trained checkpoint; shared by training and evaluation-only recovery."""
    #nfsample_list = [1,3,10]
    nfsample_list = getattr(args, 'eval_nfsamples', [1,3,10,20])

    #model.load_state_dict(save_dict['state_dict'])
    save_dict = torch.load(savepath, map_location=next(model.parameters()).device, weights_only=False)
    oodsets = list(oodloaders)
    if metrics is None:
        metrics = save_dict.get('metrics', [])
    tmp_test_path = './totalresults_{}_{}new.txt'.format(configs.dataset.trainset, configs.model.modelname)
    model.load_state_dict(save_dict['state_dict'])


    n_runs,n_splits,digits = 5,2,3
    metrics_fs = {}
    metrics_ood = {}
    auroc_by_sample = {}
    id_by_sample = {}
    for nfsample in nfsample_list:
        # Evaluate the same ID predictions against every OOD dataset.
        outs_test = eval_trainedmodel(testloader, model, q_params=list(model.parameters()),
                                      use_bayes=True, nfsample=nfsample)
        metrics_ood[nfsample] = {}
        auroc_by_sample[str(nfsample)] = {}
        for ood_name, loader in oodloaders.items():
            outs_ood = eval_trainedmodel(loader, model, q_params=list(model.parameters()),
                                         use_bayes=True, nfsample=nfsample)
            result = get_results_from_outs_without_calibration(
                outs_test, outs_ood, n_runs=n_runs, n_splits=n_splits, digits=digits)
            metrics_ood[nfsample][ood_name] = result
            auroc_by_sample[str(nfsample)][ood_name] = entropy_auroc(outs_test, outs_ood)
        # Preserve the original checkpoint field for the primary OOD dataset.
        metrics_fs[nfsample] = metrics_ood[nfsample][oodsets[0]]
        id_by_sample[str(nfsample)] = {
            name: float(value) for name, value in zip(('acc', 'nll', 'ece'), metrics_fs[nfsample][:3])
        }
        print(format_evaluation_metrics(nfsample, id_by_sample[str(nfsample)], auroc_by_sample[str(nfsample)]))

    memory_values = [row[-1] for row in metrics if row[-1] > 0]
    gpu_memory = ((float(np.mean(memory_values)), float(np.std(memory_values)))
                  if memory_values else (None, None))
    # Retain the trained checkpoint's state, optimizer, epoch, and validation metrics.
    save_dict.update(metrics=metrics, gpu_memory=gpu_memory,
                     metrics_fs=metrics_fs, metrics_ood=metrics_ood, metrics_id=id_by_sample)

    torch.save(save_dict, savepath)
    result_json = getattr(args, 'result_json', None) or savepath + '.ood.json'
    if getattr(args, 'nexp', 1) > 1 and getattr(args, 'result_json', None):
        result_json = result_json.removesuffix('.json') + f'.seed{configs.exp.seeds}.json'
    places365_root = os.path.realpath(os.environ.get('PLACES365_ROOT', '/home/yohan/Datasets/places365/val_256'))
    table_path = save_seed_results(result_json, configs, savepath, auroc_by_sample, oodsets, places365_root, id_metrics=id_by_sample,
                      history_path=getattr(args, 'result_history', None),
                      table_seeds=getattr(args, 'table_seeds', None),
                      table_nfsamples=getattr(args, 'table_nfsamples', None),
                      table_ddof=getattr(args, 'table_ddof', None) or 0,
                      table_filename=getattr(args, 'table_filename', None) or 'results_table.txt')
    print(f'Per-seed OOD results: {result_json}')
    if table_path is not None:
        print(f'Updated results table: {table_path}')
    else:
        print('Results table pending: waiting for all expected seeds and metrics.')
    msg = ' test saved at {}'.format(savepath)
    
    
    

    ## write results                        
    msg = ' test saved at {}'.format(savepath)
    write_msg(current_msg='{}'.format(savepath),tmp_logger_path=tmp_test_path)     
    msg = ''
    for sample_count in metrics_ood:
        msg += format_evaluation_metrics(sample_count, id_by_sample[str(sample_count)],
                                         auroc_by_sample[str(sample_count)]) + '\n'
    write_msg('{}'.format(msg),tmp_logger_path=tmp_test_path)     
    
        
    
    return 
    




if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='exp_fbnn')    
    parser.add_argument('--ntrainset',type=int, default=45000)       
    parser.add_argument('--nepoch',type=int, default=200)       
    

    parser.add_argument('--nexp',type=int, default=1)       
    parser.add_argument('--configs',type=str, default='abc')    
    parser.add_argument('--modelname',type=str, default='bmafbresnet18st', choices=['bmafbresnet18st', 'bmafbresnet50st'])    
    parser.add_argument('--reglambda',type=float, default=.1)    
    parser.add_argument('--eval-train', action='store_true',
                        help='also evaluate the training set; use with --reglambda 0 to reproduce the baseline')
    parser.add_argument('--scalelogstd',type=int, default=-10)      
    parser.add_argument('--nctset',type=int, default=4)        
    parser.add_argument('--nfsample',type=int, default=10, help='number of samples for forward pass')  
    
    parser.add_argument('--priorstdout',type=float, default=10.)
    parser.add_argument('--nepochratio',type=float, default=.5)
    parser.add_argument('--prior-distribution', choices=['trajectory', 'fixed'],
                        help='trajectory prior (default), or zero-mean Gaussian function prior with priorstdout scale')
    parser.add_argument('--mixalpha', type=float, default=.5)    
    #parser.add_argument('--paramtrace', type=int, default=1)
     
    parser.add_argument('--seeds',type=int, default=1111)        
    parser.add_argument('--v',type=int, default='1')    
 
        
    parser.add_argument('--oodsets', nargs='+', choices=['svhn', 'svhn2', 'lsun32', 'place365'],
                        help='OOD datasets to evaluate on the same trained checkpoint')
    parser.add_argument('--eval-nfsamples', nargs='+', type=int, default=[1, 3, 10, 20])
    parser.add_argument('--places365-root', help='directory containing Places365 validation images')
    parser.add_argument('--result-json', help='path for unrounded per-seed OOD AUROC results')
    parser.add_argument('--result-history', help='append-only JSONL history (default: beside result-json)')
    parser.add_argument('--table-seeds', nargs='+', type=int, help='expected seeds for table statistics (default: 1111 1112 1113)')
    parser.add_argument('--table-nfsamples', nargs='+', type=int, help='table sample counts (default: largest evaluated count)')
    parser.add_argument('--table-filename', choices=['results_table.txt', 'results_ablation_table.txt'],
                        help='statistics table filename (default: results_table.txt)')
    parser.add_argument('--table-ddof', type=int, choices=[0, 1], help='table standard deviation ddof (default: 0)')
    parser.add_argument('--eval-checkpoint', help='evaluate this trusted local checkpoint without training')

    args = parser.parse_args()
    if any(n < 1 for n in args.eval_nfsamples):
        parser.error('--eval-nfsamples must be positive')
    args.gentime = strftime("%m%d%H%M",gmtime()) 

    #print(args.nepoch)
    
    if args.eval_checkpoint:
        main(args)
    else:
        for k in range(args.nexp):
            main(args)
            args.seeds += 1
        
        

