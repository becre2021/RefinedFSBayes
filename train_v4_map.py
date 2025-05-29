import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import math
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
from models_deepnet_v4 import *


from math import pi
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR




def setup_randomness(random_seed= 1111):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    return 




omit_keys = ['oodset',
             'nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','nwarmupepoch','mhquantile','mhnepochprior',
             'nctset','targetacc','priorstdin','priorstdout','regcls','nmhlayer','betahyp','nepochratio',
             'Tmax','nbatch',
             'relreg','updateperiod','losstype',
             'priorstdin','usepretrain','useregweight','nepochratio','topk',
             'mixcontext',        
             'swagrank','swagper',
            ]


             


def setup_savedir(configs:AttrDict, gentime:str,omit_keys = omit_keys ):
    def setup_savepath(configs, omit_keys = omit_keys):
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


    1 if os.path.isdir(savedir) else os.makedirs(savedir)
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth.tar'     
    
    print('savepath: {}'.format(savepath))           
    return savepath


def update_configs_by_args(args):
    # get configs
    #with open("./config/{}.yaml".format(args.configs),'r') as f:
    with open("{}.yaml".format(args.configs),'r') as f:        
        configs = yaml.load(f,yaml.FullLoader)
                
    # edit configs
    args_ = vars(args)
    for i_key in args_:    
        if i_key in configs['dataset']:
            configs['dataset'][i_key] = args_[i_key]
            
        if i_key in configs['model']:
            configs['model'][i_key] = args_[i_key]
        
        if i_key in configs['train']:
            print('i_key:{}, args_[i_key]: {}'.format(i_key,args_[i_key]))
            configs['train'][i_key] = args_[i_key]
        
        if i_key in configs['exp']:
            configs['exp'][i_key] = args_[i_key]
                            
    #breakpoint()
    configs = AttrDict(configs)
    return configs





dataset_list = ['cifar10','cifar100','tinyimagenet','tinyimagenet64s']
def setup_model(configs={}):    
    #if configs.dataset.trainset in ['cifar10','cifar100']:
    if configs.dataset.trainset in ['cifar10','cifar100']:
        return _setup_model_cifar(configs)

    if configs.dataset.trainset in ['tinyimagenet','tinyimagenet64s']:
        return _setup_model_tinyimagenet(configs)






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



def _setup_model_cifar(configs={}):  

    model = _setup_model_deep(configs)        
    model = model.cuda() if torch.cuda.is_available() else model
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
                
    return model,optimizer,scheduler




def _setup_model_tinyimagenet(configs={}):  
    model = _setup_model_deep(configs)        
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr= 1. ,                                
                                momentum=0.9,
                                weight_decay=2e-4,
                                )

    nstep = configs.dataset.ntrainset //configs.train.nbatch + 1
    
    nwarmupstep = configs.train.nwarmupepoch*nstep
    nlaststep   = configs.train.nepoch*nstep
    ncosinestep = nlaststep - nwarmupstep
    
    scheduler = setup_scheduler_waramupcosdecay(optimizer ,
                                                nwarmupstep = nwarmupstep, 
                                                ncosinestep = ncosinestep, 
                                                lr_warmup= float(configs.train.lr))    
    
    #loss_fn = nn.CrossEntropyLoss()
    #print(model)    
    return model,optimizer,scheduler



## ------------------------------------------------------
## loss function
## ------------------------------------------------------

def loss_fn(outs,targets,configs):
    loss_negll = nn.CrossEntropyLoss()(outs.logits,targets)          
    if configs.train.userelreg:        
        loss_reg0 = outs.reg    
        alambda = float(configs.train.reglambda)*torch.abs(loss_negll/loss_reg0).detach().clone()
        loss_reg = alambda*loss_reg0                    
    else:
        loss_reg = float(configs.train.reglambda)*outs.reg
    loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl
    
    return loss, loss_negll,loss_reg






        
def train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler , p_params=None, configs=None):
    
    #dataloader = trainloader
    context_iter = iter(contextloader)
    loss_list,lossdata_list,lossreg_list = [],[],[]
    
    #for i, ((images, targets),(images_xc,targets_xc)) in enumerate(zip(trainloader,contextloader) ):    
    for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):    


                
        # predict class
        #outs = model.predict_train(x.cuda(), xc.cuda())                                 
        outs = model.predict_train(x.cuda(), xc.cuda(), configs=configs)                                 
            
        # compute loss 
        loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs)    

        # update param
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        scheduler.step()
        
        
        # record metrics
        loss_list.append(loss.item())        
        lossdata_list.append(loss_data.item())        
        lossreg_list.append(loss_reg.item())
        
        torch.cuda.empty_cache()        
    
    #print(loss_list)
    outs = AttrDict()
    outs.avg_loss = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg = np.array(lossreg_list).mean().round(3)    
    return outs






def eval_trainedmodel(dataloader, model,q_params = None ,topk=1, nfsample=1):
    assert q_params is not None
    pred_list,label_list,logit_list = [],[],[]    

    model.eval()    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):    
            # predict class            
            outs = model.predict_eval(x.cuda(), nfsample=nfsample)                     

            # compute performance
            label_list.append(y.cpu())
            pred_list.append(outs.pred.cpu())
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






from evaluate_v2 import (ts,
                         apply_t,
                         get_acc,
                         get_ll,
                         get_ece,
                         get_ood_auroc,      
                         get_ood_auroc2,     
                         get_entropy,              
                         )
from sklearn.model_selection import KFold


import copy


def get_results_from_outs(outs,outs_ood,n_runs=5,n_splits=2,digits=3):
    
    def _compute_logmarginalprob(logits):
        assert logits.dim() == 3
        ns,ndata,ndim = logits.shape    
        ns_log_prob = logits.log_softmax(dim=-1) #(ns,ndata,ndim)
        log_prob = torch.logsumexp(ns_log_prob, dim=0) - np.log(ns)
        return log_prob
        
        
    def _compute_diversity_ratio(outs , digits=3):
        logit_list = outs.logit_list.transpose(1,0) #(nb,ns,ndim)
        yhat = logit_list .max(dim=-1)[1]
        nfsample = yhat.shape[-1]

        static_list = []
        for i_yhat in yhat:
            i_yhat_unique,i_yhat_unique_cnt = i_yhat.unique(return_counts=True)
            top1_cnt = i_yhat_unique_cnt.topk(1).values.item()
            majority_cnt = top1_cnt
            not_majority_cnt = nfsample - top1_cnt
            static_list.append( (majority_cnt/nfsample, not_majority_cnt/ nfsample , not_majority_cnt/ majority_cnt) )  #( majority_cnt/nfsample,       not_majority_cnt/ nfsample ,   not_majority_cnt/ majority_cnt )
        
        static_list = np.array(static_list).round(3)
        return static_list.mean(axis=0).round(digits), ((1/np.sqrt(nfsample))*static_list.std(axis=0)).round(digits)
        



    targets          = outs.label_list.cpu().data.numpy()    
    targets_ood1     = outs_ood.label_list.cpu().data.numpy()
    log_preds        = _compute_logmarginalprob(outs.logit_list)   
    log_ood1_preds   = _compute_logmarginalprob(outs_ood.logit_list)
    
    # ------------------------------------------------
    # evaluate acc,nll, and, ece
    # ------------------------------------------------        
    acc_runs,nll_runs,ece_runs,aucroc1,aucroc2 = [],[],[],[],[]
    indent1,indent2 = [],[]        
    outdent1,outdent2 = [],[]
    
    for runs in range(n_runs):
        for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(log_preds)):
            train_t = ts(log_preds[tr_idx], targets[tr_idx])
            test_lp = apply_t(log_preds[te_idx], train_t)                    
            acc_runs.append( get_acc(np.exp(test_lp), targets[te_idx] ) )
            nll_runs.append( -get_ll(np.exp(test_lp), targets[te_idx]) )
            ece_runs.append( get_ece(np.exp(test_lp), targets[te_idx]) )       

            #test_lp = apply_t(log_preds[te_idx], train_t)                  
            test_lp_ood = apply_t(log_ood1_preds[te_idx], train_t)  
            
            test_lp_in_out = np.concatenate([test_lp,test_lp_ood],axis=0)
            #labels_in_out = np.array(list(np.zeros( test_lp.shape[0] )) + list(np.ones(test_lp_ood.shape[0] )))
            ood_labels = np.array(  list(np.zeros( test_lp.shape[0] )) + list(np.ones(test_lp_ood.shape[0] ) )  )

            
            # aucroc1.append( get_ood_auroc(np.exp(test_full1_lp),[],targets_fullood1[te_idx] ) )       
            aucroc1.append(  get_ood_auroc2(np.exp(test_lp_in_out ), ood_labels  ) )       
                            
            tmp_indent1 =  get_entropy( np.exp(test_lp) )
            tmp_outdent1 = get_entropy( np.exp(test_lp_ood) )
            indent1.append(  tmp_indent1  )
            outdent1.append( tmp_outdent1 )                
            
            
    # ------------------------------------------------
    # evaluate acc,nll, and, ece
    # ------------------------------------------------                        
    acc,nll,ece = np.mean(acc_runs).round(digits),np.mean(nll_runs).round(digits),np.mean(ece_runs).round(digits)                                   
    aucroc1 = np.mean(aucroc1).round(digits)
    indent1,outdent1 = np.mean(indent1).round(digits),np.mean(outdent1).round(digits)
    
    
    # evaluate functional diversity
    fdiv_in =  _compute_diversity_ratio(outs , digits=3)
    fdiv_out = _compute_diversity_ratio(outs_ood , digits=3)
    
    fdivin = fdiv_in[0][1], fdiv_in[1][1]
    fdivout = fdiv_out[0][1], fdiv_out[1][1]
    
    rfdivin = fdiv_in[0][2],fdiv_in[1][2]        #mean,variance    #relative diversity: not majority / majority
    rfdivout = fdiv_out[0][2],fdiv_out[1][2]     #mean,variance
    
    return acc,nll,ece,aucroc1,indent1,outdent1,fdivin,fdivout,rfdivin,rfdivout
    




def get_results_from_outs_without_calibration(outs,outs_ood1):
    
    
    def _compute_logmarginalprob(logits):
        #print('logit.shape: {}'.format(logits.shape))
        assert logits.dim() == 3
        ns,ndata,ndim = logits.shape    
        ns_log_prob = logits.log_softmax(dim=-1) #(ns,ndata,ndim)
        log_prob = torch.logsumexp(ns_log_prob, dim=0) - np.log(ns)
        log_prob = log_prob.log_softmax(dim=-1)

        return log_prob.cpu().data.numpy()
    

    def _compute_diversity_ratio(outs , digits=3):
        logit_list = outs.logit_list.transpose(1,0) #(nb,ns,ndim)
        yhat = logit_list .max(dim=-1)[1]
        nfsample = yhat.shape[-1]

        static_list = []
        for i_yhat in yhat:
            i_yhat_unique,i_yhat_unique_cnt = i_yhat.unique(return_counts=True)
            top1_cnt = i_yhat_unique_cnt.topk(1).values.item()
            majority_cnt = top1_cnt
            not_majority_cnt = nfsample - top1_cnt
            static_list.append( (majority_cnt/nfsample, not_majority_cnt/ nfsample , not_majority_cnt/ majority_cnt) )  #( majority_cnt/nfsample,       not_majority_cnt/ nfsample ,   not_majority_cnt/ majority_cnt )
        
        static_list = np.array(static_list).round(3)
        return static_list.mean(axis=0).round(digits), ((1/np.sqrt(nfsample))*static_list.std(axis=0)).round(digits)
            
        

    targets          = outs.label_list.cpu().data.numpy()    
    targets_ood1     = outs_ood1.label_list.cpu().data.numpy()
    log_preds        = _compute_logmarginalprob(outs.logit_list)   
    log_ood1_preds   = _compute_logmarginalprob(outs_ood1.logit_list)
    
    
    
    
    
    # ------------------------------------------------
    # evaluate acc,nll, and, ece
    # ------------------------------------------------            
            
    acc            =  get_acc( np.exp(log_preds), targets )
    nll            = -get_ll(  np.exp(log_preds), targets ) 
    ece            =  get_ece(  np.exp(log_preds), targets )    
        
    test_lp_in_out = np.concatenate([log_preds,log_ood1_preds],axis=0)
    ood_labels     = np.concatenate([np.zeros(log_preds.shape[0]),np.ones(log_ood1_preds.shape[0]) ],axis=0)             
    aucroc1        =  get_ood_auroc2(np.exp(test_lp_in_out ), ood_labels  ) 
    
    
    indent1  = get_entropy(  np.exp(log_preds)     )
    outdent1 =  get_entropy( np.exp(log_ood1_preds) )
    

    # evaluate functional diversity
    fdiv_in  = _compute_diversity_ratio(outs , digits=3)
    fdiv_out = _compute_diversity_ratio(outs_ood1 , digits=3)
    
    fdivin = fdiv_in[0][1], fdiv_in[1][1]
    fdivout = fdiv_out[0][1], fdiv_out[1][1]
    
    rfdivin = fdiv_in[0][2],fdiv_in[1][2]        
    rfdivout = fdiv_out[0][2],fdiv_out[1][2]
    
    return acc,nll,ece,aucroc1,indent1,outdent1,fdivin,fdivout,rfdivin,rfdivout



#tmp_logger_path = './training_log_basemap.txt'
tmp_logger_path = './training_log_v3_basemap.txt'
def write_msg(current_msg:str,withprint=True,tmp_logger_path=tmp_logger_path):
    if withprint:
        print(current_msg)
    
    with open(tmp_logger_path, "a") as file:
        file.write(current_msg + '\n')
        file.close()    


def main(args):

    # get and update configs
    configs = update_configs_by_args(args)
    print('update configs: {}'.format(configs))


    # set savepath
    #tmp_logger_path = './training_log_v4_{}_{}_adv_hidden_v{}.txt'.format(configs.dataset.trainset,configs.model.modelname,configs.exp.v)  #retrain
    tmp_logger_path = './v{}_trainlog_{}_{}_{}_map.txt'.format(configs.exp.v,configs.dataset.ntrainset,configs.dataset.trainset,configs.model.modelname)  #retrain


    msg = 'configs \n'
    for ith_key in configs:
        msg += 'fs={} : {} \n'.format(ith_key,configs[ith_key])
    write_msg('{}'.format(msg),tmp_logger_path=tmp_logger_path)     
    print('update configs: {}'.format(configs))
    



    # set randomness
    setup_randomness(random_seed = configs.exp.seeds)
                    
    # setup dataset
    trainloader,valloader,testloader,contextloader,oodloader  =  setup_dataset(configs)    
    #oodloader = oodlist[0]
    
    # build model 
    model,optimizer,scheduler = setup_model(configs)
    
    
    
    # set savepath
    savepath = setup_savedir(configs,gentime=args.gentime)
    write_msg(current_msg='savepath: {}'.format(savepath) , tmp_logger_path = tmp_logger_path )    
    
    
    # train model
    p_params = None
    opt_val_acc,opt_val_nll = 0.0,1e8
    metrics = []
    for i in range(1,configs.train.nepoch+1):
        t = time.time()
        #outs_train = train_epochs(trainloader,contextloader, model, optimizer, loss_fn, p_params=p_params, configs = configs)
        outs_train = train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler, p_params=p_params, configs = configs)

        elapsed = time.time() - t
        current_lr = scheduler.get_last_lr()[0]  



        # # evaluate model 
        # #if ((i % configs.exp.nfreqprint == 0) and (i>=configs.train.nepoch//2 )) or (i==1):
        # if ( i % configs.exp.nfreqprint == 0 ) or (i==1):            
            
        #nfreqprint = 5 if configs.train.usepretrain else configs.exp.nfreqprint        
        nfreqprint = 1
        if ((i % nfreqprint == 0) or (i==int(configs.train.nepoch))  or  (i ==1)):
            
            
            #with torch.no_grad():
            #outs_eval_train = eval_trainedmodel(trainloader, model, q_params = list(model.parameters()) )
            outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()) , nfsample= configs.train.nfsample)
            acc_val =  get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            nll_val = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            

            msg = '[{}/{}], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f}, \t current_lr {:.5f} '.format(i,configs.train.nepoch,elapsed,
                                                                                                                                   outs_train.avg_loss,
                                                                                                                                   outs_train.avg_lossdata,
                                                                                                                                   outs_train.avg_lossreg,
                                                                                                                                   acc_val,
                                                                                                                                   nll_val,
                                                                                                                                   current_lr)

            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val, -1))
            if opt_val_nll > nll_val:
                opt_val_nll = nll_val 
                


                save_dict = {'args':args,
                             'configs':configs,
                             'metrics':metrics,
                             'state_dict': copy.deepcopy(model).state_dict(),
                             'optimizer_dict': copy.deepcopy(optimizer).state_dict(),
                             'acc_val':acc_val,
                             'nll_val':nll_val,                             
                             'epochs':i,
                }


                torch.save(save_dict, savepath)
                msg += '\t saved'
                
            ## write msg                    
            write_msg(current_msg='{}'.format(msg) , tmp_logger_path = tmp_logger_path )    
                
    
    torch.cuda.empty_cache()
    print('\n'*3)
    
    
    
    
    




    ## --------------------------------------------------
    ## test set evaluation 
    ## --------------------------------------------------

    save_dict = torch.load(savepath)
        

    nfsample_list = [1,3,10]
    model.load_state_dict(save_dict['state_dict'])
    n_runs,n_splits,digits = 5,2,3
    metrics_fs = {}
    
    
    for nfsample in nfsample_list:
        outs_test = eval_trainedmodel(testloader, model, q_params = list(model.parameters())  ,topk=configs.train.topk, nfsample= nfsample)    
        outs_ood  = eval_trainedmodel(oodloader, model, q_params   = list(model.parameters())   ,topk=configs.train.topk, nfsample= nfsample)       
        
        
        acc,nll,ece,aucroc1,indent1,outdent1,fdivin,fdivout,rfdivin,rfdivout = get_results_from_outs_without_calibration(outs_test,outs_ood)

        metrics_fs[nfsample] = acc,nll,ece,aucroc1,indent1,outdent1,fdivin,fdivout,rfdivin,rfdivout


    save_dict = {'args':args,
                'configs':configs,
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'acc_val':acc_val,
                'nll_val':nll_val,                             
                'metrics':metrics,                
                'metrics_fs':metrics_fs,
    }


    torch.save(save_dict, savepath)
    msg = ' test saved at {}'.format(savepath)
    
    ## write msg                    
    write_msg(current_msg='{}'.format(savepath) , tmp_logger_path = tmp_logger_path ) 


    print('results')
    print(metrics_fs)


    
    return 
    




if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='exp_fbnn')    
    
    
    # varying hyperparameters
    parser.add_argument('--ntrainset',type=int, default=45000)       
    
    parser.add_argument('--nexp',type=int, default=1)       
    parser.add_argument('--configs',type=str, default='abc')    
    parser.add_argument('--modelname',type=str, default='resnet20')    
    parser.add_argument('--reglambda',type=float, default=.1)    
    parser.add_argument('--scalelogstd',type=int, default=-8)      
    parser.add_argument('--nctset',type=int, default=1)        

    parser.add_argument('--userelreg', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ## add recently
    parser.add_argument('--v',type=int, default='1')    
    
    parser.add_argument('--seeds',type=int, default=1111)        
          
    
    args = parser.parse_args()
    args.gentime = strftime("%m%d%H%M",gmtime()) 
    
    print('args: {} in main'.format(vars(args)))

    #print(args.nepoch)
    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        


## resnet20 via spectral normalization
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_base.py   --modelname res2net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --seeds 1111 --v 91 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_icmlreb_base.py    --configs  ./exp_cifar100_tinyimagenet\
#                                                                 --modelname $modelname \
#                                                                 --nexp  1 \
#                                                                 --scalelogstd -6 \
#                                                                 --reglambda $reglambda  \
#                                                                 --seeds 1111 \
#                                                                 --v $v  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_map.py    --configs  ./exp_cifar10_tinyimagenet\
#                                                                 --modelname $modelname \
#                                                                 --nexp  1 \
#                                                                 --scalelogstd -6 \
#                                                                 --reglambda $reglambda  \
#                                                                 --seeds 1111 \
#                                                                 --v $v  ;
