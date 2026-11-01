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


from datasets_vit import setup_dataset
from evaluate_v2 import get_acc,get_ll

from models_vit import _get_vit_model


from train_vit_base import (setup_randomness,
                        setup_savedir,
                        update_configs_by_args,
                        setup_model,
                        eval_trainedmodel,                        
)

from train_utils import (WarmupCosineSchedule,
                         setup_randomness,
                         setup_savedir,
                         update_configs_by_args
)


             
omit_keys = ['trainset','contextset','resolution','nfbatch','wdecay','maxgradnorm',
             'seeds','indim','outdim','hiddendim','nfreqprint',
             'ntrainset','nwarmupepoch','mhquantile','mhnepochprior',
             'nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','ntrainset','nwarmupepoch','nepoch','nbatch',
             'mhquantile','mhnepochprior','targetacc','nmhlayer',
             'userelreg','regcls','betahyp','updateperiod',
             'priorstdin','usepretrain','useregweight',
             'losstype','pretrained',
             'nvalbatch','topk','usepretrain','temppeak','mixself','paramtrace','prescale',
             #'nepochratio','topk',
             ]




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

    loss_fn    = negloss_elbo            
    loss_negll = loss_fn(outs,targets)            
    loss_reg   = float(configs.train.reglambda)*outs.reg 
    
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





def  train_epochs_with_prior(trainloader,contextloader, model, optimizer, loss_fn, scheduler, update_h =False, configs=None, use_bayes=False, updated_p_params =None):

    # ## setup beta dist
    # beta_dist = Beta(.5,.5, validate_args=None)                
                
    ## due to large memory
    optimizer.zero_grad()    
    nstep_grad = int(configs.train.nfbatch // configs.train.nbatch)
    print('nstep_grad: {}'.format(nstep_grad))
     
    tmp_loss,loss_list,lossdata_list,lossreg_list = [],[],[],[]
    
    for i, (x, y) in enumerate(trainloader):    
       

        logits_samples, h_x = model.compute_logit_samples(x.cuda(), 
                                                          y=y.cuda(),
                                                          use_bayes=use_bayes, 
                                                          update_h = update_h,
                                                          configs=configs)  
            
                                   
        if use_bayes and (float(configs.train.reglambda)>0) :
            # compute sup kl                
        

            ## build adversial examples as h_x_contextset
            # build adv context on h
            h_x_copy, h_x_grad           = model.compute_h_x_grad(h_x)
            h_xc                         = fgsm_attack(h_x_copy, h_x_grad, epsilon= configs.train.mixalpha).detach().clone()                                           
            params_qf, params_pf, weight = model.compute_fdist_from_h(h_xc, p_params =updated_p_params, configs=configs) 




            # compute sup kl from h
            sup_kl = model.compute_sup_kl(params_qf, params_pf, weight=weight, Lowrank=False, Diag=True, nctset=configs.train.nctset )   
            

        else:
            sup_kl = torch.zeros(1).to(logits_samples.device)
        

        outs = AttrDict()
        outs.logits = logits_samples
        outs.reg = sup_kl                               



        loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs ,use_bayes=use_bayes)
        loss      = loss      / nstep_grad      if nstep_grad > 1 else loss
        loss_data = loss_data / nstep_grad      if nstep_grad > 1 else loss_data
        loss_reg  = loss_reg  / nstep_grad      if nstep_grad > 1 else loss_reg
        
        # get gradient 
        loss.backward()

        tmp_loss.append( (loss.item(),loss_data.item(),loss_reg.item() ) )
        if (i+1) % nstep_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           configs.train.maxgradnorm)

            # update param
            optimizer.step()                
            scheduler.step()

            optimizer.zero_grad()
            
            
            tmp_loss = np.array(tmp_loss).sum(axis=0)
            loss_list.append( tmp_loss[0] )        
            lossdata_list.append( tmp_loss[1] )        
            lossreg_list.append(  tmp_loss[2] )   
            tmp_loss = []
        

        # record metrics
        loss_list.append(loss.item())        
        lossdata_list.append(loss_data.item())        
        lossreg_list.append(loss_reg.item())
        
        # empty cuda
        torch.cuda.empty_cache()        
        
        
    outs = AttrDict()
    outs.avg_loss     = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg  = np.array(lossreg_list).mean().round(3)  
    outs.nstep        = int((i+1)//nstep_grad)      
    return outs




def eval_trainedmodel(dataloader, model,q_params = None,use_bayes=True , nfsample=1 ):
    assert q_params is not None
    pred_list,label_list = [],[]    
    logit_diversity_list = []

    model.eval()    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):    
            # predict class            
            outs = model.predict_eval(x.cuda(),use_bayes=use_bayes, nfsample=nfsample)                     

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






def process_priorparamlist_as_swagdiag(prior_parameters_list,configs=None, eps_log=1e-8, eps_clamp=1e-20):
    assert len(prior_parameters_list) >= 1
    nprior_model = len(prior_parameters_list)

    emp_first_moment   = [torch.zeros_like(ith_param).to(ith_param) for ith_param in prior_parameters_list[0]]
    emp_second_moment  = [torch.zeros_like(ith_param).to(ith_param) for ith_param in prior_parameters_list[0]]    
    
    for current_path_list in prior_parameters_list:
        for ith_first_moment, ith_second_moment, ith_current in zip(emp_first_moment,emp_second_moment,current_path_list):
            ith_first_moment.data  += (1/nprior_model)*ith_current.data
            ith_second_moment.data += (1/nprior_model)*torch.pow(ith_current.data,2)

    updated_p_logstd = [torch.log( torch.sqrt( torch.clamp(ith_second_moment - ith_first_moment**2, min=eps_clamp)) ).to(ith_first_moment) for ith_first_moment,ith_second_moment in zip(emp_first_moment,emp_second_moment)]
    updated_p_mean   = emp_first_moment    
    
    updated_p_params = (updated_p_mean,updated_p_logstd)
    return updated_p_params








tmp_logger_path = './training_log_proposed'

def write_msg(current_msg:str,withprint=True,tmp_logger_path=tmp_logger_path):
    if withprint:
        print(current_msg)
    
    with open(tmp_logger_path, "a") as file:
        file.write(current_msg + '\n')
        file.close()    










## reference: https://github.com/jeonsworld/ViT-pytorch/blob/460a162767de1722a014ed2261463dbbc01196b6/train.py#L18

def main(args):
    use_bayes=False
    p_mean_weight = None
                                
    ## get and update configs
    configs = update_configs_by_args(args)
    print('update configs: {}'.format(configs))


    steps_per_epochs    =  (configs.dataset.ntrainset // configs.train.nfbatch)        
    total_trainepochs   =  (configs.train.nstep // steps_per_epochs) + 1            

    epochs_for_bayes             = int(total_trainepochs*configs.train.nepochratio)
    past_periods                 =  int(total_trainepochs*0.1)
    ntrajectory                  = 4
    epochs_for_priorconstruction = np.linspace( epochs_for_bayes - past_periods, epochs_for_bayes, ntrajectory,dtype=int) 


    nfreqprint                   = configs.exp.nfreqprint            



    print('-'*100)
    print('total_nepochs:{}, \t epochs_for_bayes:{}, \t epochs_for_priorconstruction: {}'.format(total_trainepochs,epochs_for_bayes,epochs_for_priorconstruction))
    print('-'*100)



    ## set randomness
    setup_randomness(random_seed = configs.exp.seeds)                         
                    
    ## setup dataset
    trainloader,valloader, _, contextloader, _   =  setup_dataset(configs)    
    
    
    ## build model 
    model,optimizer,scheduler = setup_model(configs)
    print('priorstdout:{}'.format(model.priorstdout))
    
    ## set savepath
    savepath = setup_savedir(configs,gentime=args.gentime,omit_keys=omit_keys)
    
    #tmp_logger_path = './training_log_{}_{}_v{}.txt'.format(configs.dataset.trainset,configs.model.infer,configs.exp.v)    
    tmp_logger_path = './training_log_{}_{}_v{}.txt'.format(configs.model.infer,configs.dataset.trainset,configs.exp.v)    

    write_msg(current_msg='savepath: {}'.format(savepath),tmp_logger_path=tmp_logger_path)    




    ## train model
    p_params = None
    opt_val_acc,opt_val_nll = 0.0,1e8
    metrics = []
    accumulated_nstep = 0    
    shouldtrain = True
    prior_param_list = []
    updated_p_params = None
    use_bayes = False

    count_epoch = 0
    while shouldtrain:               
        
        ## set prior params       
        if  (count_epoch == epochs_for_bayes)  and (not configs.train.usepretrain):                                    

            updated_p_params = process_priorparamlist_as_swagdiag(prior_param_list,configs=configs)
            model.update_params_p(*updated_p_params)
            use_bayes=True
            

            ## save pretrained model            
            save_dict = {'args':args,
                        'configs':configs,
                        'state_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'prior_param_list' : prior_param_list,
                        'updated_p_params' : updated_p_params,
                        'nepochs_pretrain' : epochs_for_bayes,
            }
            savepath_now = './results/vit_pretrained_at_{}nepoch_{}.pth.tar'.format( configs.train.nepochratio,configs.train.losstype )
            print('save pretrained model at {}'.format(savepath_now))
            torch.save(save_dict, savepath_now)

            ## set trainloader small due to context set
            #trainloader = trainloader_small



        # train model            
        t = time.time()            
        outs_train = train_epochs_with_prior(trainloader,
                                             contextloader, 
                                             model, 
                                             optimizer,
                                             loss_fn,
                                             scheduler,
                                             configs          = configs, 
                                             use_bayes        = use_bayes,
                                             updated_p_params = updated_p_params,
                                             update_h         = True if count_epoch in epochs_for_priorconstruction else False,
                                             )
        accumulated_nstep += outs_train.nstep
        elapsed = time.time() - t
        
        
        if count_epoch in epochs_for_priorconstruction:
            prior_param_list.append(model.get_current_p_params())
        
        
        # count epoch
        count_epoch +=1                
        
        
        # evaluate the trained model on validation set
        nfreqprint = configs.exp.nfreqprint        
        #nfreqprint = 1                         
        nmultiple = int(accumulated_nstep  / outs_train.nstep)        
        if  (nmultiple % nfreqprint) == 0 or ( nmultiple == 1) or (accumulated_nstep > configs.train.nstep) :        
            
            outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()), use_bayes=use_bayes, nfsample= configs.train.nfsample)

            acc_val       =  get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            nll_val       = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
                
            
            
            msg = '[{}/{} steps, {}/{} epochs], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f}, \t usebayes: {} '.format(accumulated_nstep,
                                                                                                                                    configs.train.nstep,
                                                                                                                                    count_epoch,
                                                                                                                                    total_trainepochs,
                                                                                                                                    elapsed,
                                                                                                                                    outs_train.avg_loss,
                                                                                                                                    outs_train.avg_lossdata,
                                                                                                                                    outs_train.avg_lossreg,
                                                                                                                                    acc_val,
                                                                                                                                    nll_val,
                                                                                                                                    use_bayes)

            

            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val))
            
            
            if (opt_val_nll > nll_val) and use_bayes:
            
                opt_val_nll = nll_val                 
      
                save_dict = {'args':args,
                             'configs':configs,
                             'state_dict': model.state_dict(),
                             'optimizer_dict': model.state_dict(),
                             'acc_val':acc_val,
                             'nll_val':nll_val,      
                             'accm_nstep':accumulated_nstep,
                }                
                torch.save(save_dict, savepath)
                msg += '\t saved'
                
            ## write msg                    
            write_msg(current_msg='{}'.format(msg),tmp_logger_path=tmp_logger_path)                    



    
        if accumulated_nstep > configs.train.nstep:
            print('total_nstep: {}, nstep: {}'.format(accumulated_nstep, configs.train.nstep))
            print('early stopping for training')
            shouldtrain = False 
            print('\n'*3)
            break        
        torch.cuda.empty_cache()
        
        
        
    
    
    #print('\n'*3)
    return 



if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='exp_fbnn')    

    parser.add_argument('--nexp',type=int, default=1)       
    parser.add_argument('--configs',type=str, default='abc')    
    parser.add_argument('--modelname',type=str, default='vitbp16r224')    
    parser.add_argument('--reglambda',type=float, default=.1)    
    parser.add_argument('--scalelogstd',type=int, default=-4)      
    parser.add_argument('--nctset',type=int, default=1)        
    
    #parser.add_argument('--userelreg', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ##add recently 05/03
    parser.add_argument('--infer',type=str, default='vbp', choices=['vbp'])        
    parser.add_argument('--nstep',type=int, default=10)    
    parser.add_argument('--nfsample',type=int, default=1, help='number of samples for forward pass')  
    
    
    parser.add_argument('--nepochratio',type=float, default=.5)    
    parser.add_argument('--priorstdout',type=float, default=10.)
 
    parser.add_argument('--losstype',type=str, default='elbo')    
    parser.add_argument('--nbatch',type=int, default=64)    
 

    parser.add_argument('--mixalpha',type=float, default=.1)        
    
    
    parser.add_argument('--paramtrace',type=int, default=0)
    
    parser.add_argument('--v',type=int, default='1')        
    parser.add_argument('--seeds',type=int, default=1111)        
          
    
    parser.add_argument('--lr',type=float, default=1e-2)
    

        
    args = parser.parse_args()
    args.gentime = strftime("%m%d%H%M",gmtime()) 

    #print(args.nepoch)
    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        

