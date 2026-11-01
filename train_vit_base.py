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

from train_utils import (WarmupCosineSchedule,
                         setup_randomness,
                         setup_savedir,
                         update_configs_by_args
)

omit_keys = ['trainset','contextset','resolution','nfbatch','wdecay','nepochratio','maxgradnorm','nepoch',
             'nperiod','seeds','indim','outdim','hiddendim','nfreqprint',
             'ntrainset','nwarmupepoch','mhquantile','mhnepochprior',
             'nctset','targetacc','priorstdin','priorstdout','regcls','nmhlayer','betahyp',
             'nvalbatch',
             'topk',
             'losstype',
             'oodset','scalelogstd','temppeak','nbatch','updateperiod','mixself','mixalpha','usepretrain',
            ]



#from models_vit import _get_vit_model
from models_vit import _get_vit_model


def setup_model(configs: AttrDict):
    model = _get_vit_model(configs)    
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr= float(configs.train.lr) ,
                                momentum=0.9,
                                weight_decay=configs.train.wdecay,
                                )
    
    if configs.dataset.trainset == 'cifar100':
        warmupsteps = 500
        t_total     = 10000

    if configs.dataset.trainset in ['flowers102','dtd','aircraft','pets37','caltech101']:
        
        t_total = int(configs.train.nstep)
        warmupsteps = int(0.1*t_total) 
        
    scheduler = WarmupCosineSchedule(optimizer, 
                                      warmup_steps=warmupsteps,
                                      t_total=t_total)
    
    print('-'*100)
    print('warmupstes: {}, total_trainingsteps: {}'.format(warmupsteps,t_total))
    print('-'*100)
    return model,optimizer,scheduler





## ------------------------------------------------------
## loss function
## ------------------------------------------------------

def loss_fn(outs,targets,configs):
    loss_negll = nn.CrossEntropyLoss()(outs.logits,targets)          
    loss_reg = float(configs.train.reglambda)*outs.reg
    loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl
    
    return loss, loss_negll, loss_reg




        
tmp_logger_path = './training_log_basemap.txt'
def write_msg(current_msg:str,withprint=True,tmp_logger_path=tmp_logger_path):
    if withprint:
        print(current_msg)
    
    with open(tmp_logger_path, "a") as file:
        file.write(current_msg + '\n')
        file.close()    



#def train_epochs(trainloader,contextloader, model, optimizer, loss_fn, p_params=None, configs=None):
def train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler , p_params=None, configs=None):
    
    #dataloader = trainloader
    context_iter = iter(contextloader)
    loss_list,lossdata_list,lossreg_list = [],[],[]
    
    ## due to large memory
    optimizer.zero_grad()    
    nstep_grad = int(configs.train.nfbatch // configs.train.nbatch)

    
    tmp_loss = []    
    for i, (x, y) in enumerate(trainloader):    

        # predict class
        outs = model.predict_train(x.cuda(), None, configs=configs)                                             
        # compute loss 
        loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs)    
        loss      = loss / nstep_grad if nstep_grad > 1 else loss
        loss_data = loss_data / nstep_grad if nstep_grad > 1 else loss_data
        loss_reg  = loss_reg / nstep_grad if nstep_grad > 1 else loss_reg
        #breakpoint()
        
        loss.backward()

        tmp_loss.append( (loss.item(),loss_data.item(),loss_reg.item() ) )
        if (i+1) % nstep_grad == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), configs.train.maxgradnorm)

            #breakpoint()
            
            # update param
            scheduler.step()
            optimizer.step()                
            optimizer.zero_grad()
            
            
            tmp_loss = np.array(tmp_loss).sum(axis=0)
            #print(tmp_loss)
            # record metrics
            loss_list.append( tmp_loss[0] )        
            lossdata_list.append( tmp_loss[1] )        
            lossreg_list.append(  tmp_loss[2] )   
            tmp_loss = []
        
        torch.cuda.empty_cache()        
    
    #print(loss_list)
    outs = AttrDict()
    outs.avg_loss = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg = np.array(lossreg_list).mean().round(3)    
    outs.nstep = int((i+1)//nstep_grad)
    return outs



def eval_trainedmodel(dataloader, model,q_params = None , nfsample=1):
    assert q_params is not None
    pred_list,label_list = [],[]    

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
        
    # evaluate model
    outs = AttrDict()
    outs.pred_list = pred_list
    outs.label_list = label_list    
    return outs    




## reference: https://github.com/jeonsworld/ViT-pytorch/blob/460a162767de1722a014ed2261463dbbc01196b6/train.py#L18

def main(args):

    # get and update configs
    configs = update_configs_by_args(args)
    if (configs.train.nbatch < 1 or configs.train.nfbatch < configs.train.nbatch
            or configs.train.nfbatch % configs.train.nbatch):
        raise ValueError('Effective batch must be a positive multiple of microbatch size')
    print('update configs: {}'.format(configs))

    # set randomness
    setup_randomness(random_seed = configs.exp.seeds)
                    
    # setup dataset
    #breakpoint()
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

        
    accumulated_nstep = 0    
    shouldtrain = True
    while shouldtrain:
        t = time.time()

        outs_train = train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler, p_params=p_params, configs = configs)
        accumulated_nstep += outs_train.nstep
        elapsed = time.time() - t


        current_lr = scheduler.get_last_lr()[0]


        nmultiple = int(accumulated_nstep  / outs_train.nstep)
        if  (nmultiple % configs.exp.nfreqprint) == 0 or ( nmultiple == 1) or (accumulated_nstep > configs.train.nstep) :            
            
            outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()) , nfsample= configs.train.nfsample)
            acc_val = get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            nll_val = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())

            msg = '[{}/{} steps], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f} , \t current_lr {:.5f} '.format(accumulated_nstep,
                                                                                                                                        configs.train.nstep,
                                                                                                                                        elapsed,
                                                                                                                                        outs_train.avg_loss,
                                                                                                                                        outs_train.avg_lossdata,
                                                                                                                                        outs_train.avg_lossreg,
                                                                                                                                        acc_val,
                                                                                                                                        nll_val,
                                                                                                                                        current_lr)

            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val))
            
            
            # if val_acc < outs_eval_val.acc:
            #     val_acc = outs_eval_val.acc            
            if opt_val_nll > nll_val:
                opt_val_nll = nll_val 
                
                save_dict = {'args':args,
                             'configs':configs,
                             'metrics':metrics,
                             'state_dict': model.state_dict(),
                             'optimizer_dict': model.state_dict(),
                             'acc_val':acc_val,
                             'nll_val':nll_val,                             
                             'accm_nstep':accumulated_nstep,
                }
                torch.save(save_dict, savepath)
                msg += '\t saved'
            
            ## write msg                    
            write_msg(current_msg='{}'.format(msg))    
                
    
    
        if accumulated_nstep > configs.train.nstep:
            print('total_nstep: {}, nstep: {}'.format(accumulated_nstep, configs.train.nstep))
            print('early stopping for training')
            shouldtrain = False 
            print('\n'*3)            
            break    
        torch.cuda.empty_cache()
    
    
    
    return 
    




if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='exp_fbnn')
    parser.add_argument('--nfbatch', type=int, default=argparse.SUPPRESS,
                        help='Effective training batch size after gradient accumulation')
    parser.add_argument('--nbatch', type=int, default=argparse.SUPPRESS,
                        help='Training microbatch size')    
    
    # varying hyperparameters
    parser.add_argument('--nexp',type=int, default=1)       
    parser.add_argument('--configs',type=str, default='abc')    
    parser.add_argument('--modelname',type=str, default='vitbp16r224')    
    parser.add_argument('--reglambda',type=float, default=.1)    
    parser.add_argument('--scalelogstd',type=int, default=-8)      
    parser.add_argument('--nctset',type=int, default=1)        
    
    #parser.add_argument('--userelreg', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ## add recently
    parser.add_argument('--nfsample',type=int, default=1)    
    parser.add_argument('--v',type=int, default='1')    
    parser.add_argument('--infer',type=str, default='map', choices=['map'])        
    parser.add_argument('--nstep',type=int, default=10)    
    
    parser.add_argument('--seeds',type=int, default=1111)        
          
    
    args = parser.parse_args()
    args.gentime = strftime("%m%d%H%M",gmtime()) 
    
    print('args: {} in main'.format(vars(args)))

    #print(args.nepoch)
    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        

