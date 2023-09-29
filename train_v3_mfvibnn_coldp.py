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



#omit_keys = ['nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint']
#omit_keys = ['nperiod','seeds','in_dim','out_dim','hidden_dim','nfreqprint','ntrainset','nwarmupepoch','mhquantile','mhnepochprior','targetacc']
omit_keys = ['nperiod','Tmax','seeds','in_dim','out_dim','hidden_dim','nfreqprint','ntrainset','nwarmupepoch','mhquantile','mhnepochprior','targetacc',
             'priorstdin','priorstdout',
             'relreg','updateperiod','losstype',
             'usepretrain','useregweight','nepochratio','topk',
             'mixcontext','nmhlayer','betahyp','nepoch','nbatch','nctset','regcls','userelreg',   
             ]

             
             
def setup_savedir(configs:AttrDict, gentime:str ):
    
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
    #savedir = './results/{}_smalltrain{}/{}/'.format(configs.dataset.trainset,smallsetratio,set_savepath(args,configs))
    1 if os.path.isdir(savedir) else os.makedirs(savedir)
    #savepath = savedir + 'seeds{}_gentime{}'.format(str(args.seeds),str(args.gentime)) + '.pth'
    #savepath = savedir + 'seeds{}_gentime{}'.format(str(args.seeds),str(args.gentime)) + '.pth'        
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth'     
    
    print('savepath: {}'.format(savepath))           
    return savepath

def loss_fn(outs,targets,configs):
    #loss_negll = nn.CrossEntropyLoss()(outs.logits,targets)          
    
    
    def negloss_elbo(outs,targets):
        #set loss
        loss_fun = nn.CrossEntropyLoss(reduction='none')    
        
        #compute loss        
        num_sample,nb,nout = outs.logits.shape
        y_samples = targets.view(1,-1).repeat(num_sample,1) #(num_sample,nb)
        neglog_y_given_f = loss_fun(outs.logits.reshape(num_sample*nb,-1), y_samples.reshape(-1))
        return neglog_y_given_f.mean()
        
    
    # loss_negll = negloss_elbo(outs,targets)        
    # loss_reg0 = outs.reg
    # alambda = float(configs.train.reglambda)*torch.abs(loss_negll/loss_reg0).detach().clone()
    # loss_reg = alambda*loss_reg0
    # loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl

    loss_negll = negloss_elbo(outs,targets)        
    
    # if configs.train.userelreg:        
    #     loss_reg0 = outs.reg    
    #     alambda = float(configs.train.reglambda)*torch.abs(loss_negll/loss_reg0).detach().clone()
    #     loss_reg = alambda*loss_reg0                    
    #     #print('check relative reglambda')
    # else:
    #     loss_reg = float(configs.train.reglambda)*outs.reg
    # loss = loss_negll + loss_reg   # -(loss_data - loss_kl) = loss_negdata + loss_kl

    loss_reg = float(configs.train.reglambda)*outs.reg
    loss = loss_negll + loss_reg   # -(loss_data - loss_kl
    #print('loss_negll {}, loss_reg {}'.format(loss_negll, loss_reg))
    return loss, loss_negll, loss_reg




# def train_epochs(trainloader,contextloader, model, optimizer, loss_fn, scheduler, p_params=None, configs=None):

    
#     loss_list,lossdata_list,lossreg_list = [],[],[]
#     for i, ((x, y),(xc,yc)) in enumerate(zip(trainloader,contextloader) ):    
        
#         # predict class
#         #outs = model.predict_train(x.cuda(), xc.cuda(), nctset=configs.train.nctset)                                 
#         # outs = model.predict_train(x.cuda(), 
#         #                            xc.cuda(), 
#         #                            nctset= configs.train.nctset,
#         #                            reglambda=configs.train.reglambda)                                                 
#         outs = model.predict_train(x.cuda(),xc.cuda(),configs)
            
#         # compute loss 
#         #loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs )    
#         loss, loss_data, loss_reg = loss_fn(outs, y.cuda(), configs , model)    

#         # update param
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()    
#         scheduler.step()
        

#         # record metrics
#         loss_list.append(loss.item())        
#         lossdata_list.append(loss_data.item())        
#         lossreg_list.append(loss_reg.item())
        
#         torch.cuda.empty_cache()        
    
#     #print(loss_list)
#     outs = AttrDict()
#     outs.avg_loss = np.array(loss_list).mean().round(3)
#     outs.avg_lossdata = np.array(lossdata_list).mean().round(3)
#     outs.avg_lossreg = np.array(lossreg_list).mean().round(3)    
#     return outs



# def eval_trainedmodel(dataloader, model,q_params = None , nfsample=10):
#     assert q_params is not None
    
#     pred_list,label_list = [],[]    
    
#     with torch.no_grad():
#         for i, (x, y) in enumerate(dataloader):    
#             # predict class            
#             outs = model.predict_eval(x.cuda(), nfsample=nfsample)                     

#             # compute performance
#             pred_list.append(outs.pred.cpu())
#             label_list.append(y.cpu())

#             torch.cuda.empty_cache() 
    
#     pred_list = torch.cat(pred_list,dim=0)    
#     label_list = torch.cat(label_list,dim=0)    
        
#     # evaluate model
#     outs = AttrDict()
#     outs.pred_list = pred_list
#     outs.label_list = label_list    
#     return outs    


#tmp_logger_path = './training_log_mfvibnn.txt'
tmp_logger_path = './training_log_v3_mfvibnn.txt'

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
        #scheduler.step()
        
        

        #with torch.no_grad():
        #outs_eval_train = eval_trainedmodel(trainloader, model, q_params = list(model.parameters()) )
        #outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()) )
        outs_eval_val = eval_trainedmodel(valloader, model , q_params = list(model.parameters()) , nfsample= configs.train.nfsample)
        
        acc_val =  get_acc(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())
        nll_val = -get_ll(outs_eval_val.pred_list.cpu().data.numpy(),outs_eval_val.label_list.cpu().data.numpy())


        # evaluate model 
        if (i % configs.exp.nfreqprint == 0) or (i==1):
            msg = '[{}/{}], taken (sec): {:.3f}  : loss {:.3f}, \t lossdata {:.3f}, \t  lossreg {:.3f}, \t acc_val {:.3f}, \t nll_val {:.3f} '.format(i,configs.train.nepoch,elapsed,
                                                                                                                                   outs_train.avg_loss,
                                                                                                                                   outs_train.avg_lossdata,
                                                                                                                                   outs_train.avg_lossreg,
                                                                                                                                   acc_val,
                                                                                                                                   nll_val)

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
                             'epochs':i,
                }
                torch.save(save_dict, savepath)
                msg += '\t saved'                
                
                
            ## write msg                    
            write_msg(current_msg='{}'.format(msg))    

   
    
    torch.cuda.empty_cache()
    print('\n'*3)
    write_msg(current_msg='done \n\n\n')    
    
        
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
    
    parser.add_argument('--userelreg', action=argparse.BooleanOptionalAction, default=False, help='relative reglambda for regularization') ## add recently
    parser.add_argument('--v',type=int, default='1')    
    
    parser.add_argument('--seeds',type=int, default=1111)        
        
    args = parser.parse_args()
    args.gentime = strftime("%m%d%H%M",gmtime()) 

    #print(args.nepoch)
    
    for k in range(args.nexp):
        main(args)
        args.seeds += 1
        
        

