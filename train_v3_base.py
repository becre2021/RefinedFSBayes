
import yaml
import random
import numpy as np
import math
import os
import argparse
import time
from time import gmtime, strftime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from collections.abc import Mapping
from torchvision import datasets, transforms
from attrdict import AttrDict

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR


from datasets_v2 import setup_dataset
from models_deepnet_v3 import *
from evaluate_v2 import get_acc,get_ll




def setup_randomness(random_seed= 1111):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    return 




omit_keys = ['nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','ntrainset','nwarmupepoch','mhquantile','mhnepochprior',
             'nctset','targetacc','priorstdin','priorstdout','regcls','nmhlayer','betahyp','nepochratio',
             'Tmax','nbatch',
             'relreg','updateperiod','losstype',
             'priorstdin','usepretrain','useregweight','nepochratio','topk',
             'mixcontext',        
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
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth'     
    
    print('savepath: {}'.format(savepath))           
    return savepath


def update_configs_by_args(args):
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


def setup_model(configs: AttrDict) -> nn.Module:    
    
    if configs.dataset.trainset in ['cifar10','cifar100']:
        return _setup_model_cifar(configs)

def _setup_model_fmnist(configs: AttrDict) -> nn.Module:
    if configs.model.modelname in model_deep_list:
        print('get deep model: {}'.format(configs.model.modelname))      
        model = _setup_model_deep(configs)
        
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr= float(configs.train.lr) ,
                                momentum=0.9,
                                weight_decay=5e-4,
                                )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           float(configs.train.Tmax),
                                                           eta_min=1e-5, 
                                                           last_epoch=- 1, verbose=False
                                                           )
        
    return model,optimizer,scheduler


def _setup_model_cifar(configs: AttrDict) -> nn.Module:

    if configs.model.modelname in model_deep_list:
        print('get deep model: {}'.format(configs.model.modelname))      
        model = _setup_model_deep(configs)
        
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.SGD(model.parameters(), 
                                #lr= float(configs.train.lr) ,
                                lr= 1. ,                                
                                momentum=0.9,
                                weight_decay=5e-4,
                                )
    
    
    try:
        nstep = configs.dataset.ntrainset //configs.train.nbatch + 1
        nwarmupstep = configs.train.nwarmupepoch*nstep
        nlaststep   = configs.train.nepoch*nstep
        ncosinestep = nlaststep - nwarmupstep
    except:
        nstep = 1
        nwarmupstep = 1
        nlaststep = 2 
        ncosinestep = nlaststep - nwarmupstep
    #print('nstep: {}, nwarmupstep: {}, ncosinestep: {}'.format(nstep,nwarmupstep,ncosinestep))
    
    scheduler = setup_scheduler_waramupcosdecay(optimizer ,
                                                nwarmupstep = nwarmupstep, 
                                                ncosinestep = ncosinestep, 
                                                lr_warmup= float(configs.train.lr))    
    
    return model,optimizer,scheduler




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
    
    context_iter = iter(contextloader)
    loss_list,lossdata_list,lossreg_list = [],[],[]


    for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):            
                
        # predict class
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
    
    outs = AttrDict()
    outs.avg_loss = np.array(loss_list).mean().round(3)
    outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
    outs.avg_lossreg = np.array(lossreg_list).mean().round(3)    
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

    # set randomness
    setup_randomness(random_seed = configs.exp.seeds)
                    
    # setup dataset
    trainloader,valloader, _, contextloader, _   =  setup_dataset(configs)    
    
    # build model 
    model,optimizer,scheduler = setup_model(configs)
    
    # set savepath
    savepath = setup_savedir(configs,gentime=args.gentime)
    write_msg(current_msg='savepath: {}'.format(savepath))    
    
    
    # train model
    p_params = None
    opt_val_acc,opt_val_nll = 0.0,1e8
    metrics = []
    for i in range(1,configs.train.nepoch+1):
        t = time.time()
        outs_train = train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler, p_params=p_params, configs = configs)
        elapsed = time.time() - t


        # evaluate model 
        if ( i % configs.exp.nfreqprint == 0 ) or (i==1):            

            outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()) , nfsample= configs.train.nfsample)
            acc_val = get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
            nll_val = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())

            
            msg = '[{}/{}], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f} '.format(i,configs.train.nepoch,elapsed,
                                                                                                                                   outs_train.avg_loss,
                                                                                                                                   outs_train.avg_lossdata,
                                                                                                                                   outs_train.avg_lossreg,
                                                                                                                                   acc_val,
                                                                                                                                   nll_val)

            metrics.append((elapsed,outs_train.avg_loss,outs_train.avg_lossdata,outs_train.avg_lossreg, acc_val, nll_val))
            
            
            if opt_val_nll > nll_val:
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
    
    
    # varying hyperparameters
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

    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        


