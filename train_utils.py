import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from torchvision import datasets, transforms


#from attrdict import AttrDict
from addict import Dict as AttrDict


import os
from time import gmtime, strftime


import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR



class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    
    
    
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
             'nctset','targetacc','priorstdin','priorstdout','regcls','nmhlayer','betahyp',
            ]


# def setup_savedir(configs:AttrDict, gentime:str,omit_keys = omit_keys ):
#     def setup_savepath(configs, omit_keys = omit_keys):
#         #omit_keys = ['nperiod']
#         full_name = ''
#         for ith_key in configs:
#             tmp_name = ''
#             for jth_key in configs[ith_key]:
#                 if jth_key not in omit_keys:
#                     tmp_name+='{}{}_'.format(jth_key,configs[ith_key][jth_key])
#                 else:
#                     pass
#             #print(tmp_name)
#             full_name += tmp_name        
#         return full_name[:-1]
    
#     #savedir = './results/{}/{}/'.format(configs.dataset.trainset,setup_savepath(configs))
#     savedir = './results/{}_{}/{}/'.format(configs.dataset.trainset,configs.dataset.contextset,setup_savepath(configs))
#     #1 if os.path.isdir(savedir) else os.makedirs(savedir)
#     os.makedirs(savedir,exist_ok=True)
    
#     savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth.tar'     
    
#     print('savepath: {}'.format(savepath))           
#     return savepath



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
    
    

    os.makedirs(savedir, exist_ok=True)
    savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth.tar'     
    
    print('savepath: {}'.format(savepath))           
    return savepath




# def setup_savedir_v2(configs:AttrDict, gentime:str,omit_keys = omit_keys ):
#     def setup_savepath(configs, omit_keys = omit_keys):
#         #omit_keys = ['nperiod']
#         full_name = ''
#         for ith_key in configs:
#             tmp_name = ''
#             for jth_key in configs[ith_key]:
#                 if jth_key not in omit_keys:
#                     tmp_name+='{}{}_'.format(jth_key,configs[ith_key][jth_key])
#                 else:
#                     pass
#             #print(tmp_name)
#             full_name += tmp_name        
#         return full_name[:-1]
    
#     #savedir = './results/{}/{}/'.format(configs.dataset.trainset,setup_savepath(configs))
#     savedir = './results/{}_{}/{}/'.format(configs.dataset.trainset,configs.dataset.contextset,setup_savepath(configs))
#     #1 if os.path.isdir(savedir) else os.makedirs(savedir)
#     os.makedirs(savedir,exist_ok=True)
    
#     #savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime)) + '.pth.tar'     
#     savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime))      
    
#     print('savepath: {}'.format(savepath))           
#     return savepath


# def setup_savedir_v2(configs:AttrDict, gentime:str,omit_keys = omit_keys ):
#     def setup_savepath(configs, omit_keys = omit_keys):
#         #omit_keys = ['nperiod']
#         full_name = ''
#         for ith_key in configs:
#             tmp_name = ''
#             for jth_key in configs[ith_key]:
#                 if jth_key not in omit_keys:
#                     tmp_name+='{}{}_'.format(jth_key,configs[ith_key][jth_key])
#                 else:
#                     pass
#             #print(tmp_name)
#             full_name += tmp_name        
#         return full_name[:-1]
    
#     #savedir = './results/{}/{}/'.format(configs.dataset.trainset,setup_savepath(configs))
#     savedir = './results/{}_{}/preswag_{}/'.format(configs.dataset.trainset,configs.dataset.contextset,setup_savepath(configs))
#     #1 if os.path.isdir(savedir) else os.makedirs(savedir)
#     os.makedirs(savedir,exist_ok=True)
    
#     savepath = savedir + 'seeds{}_gentime{}'.format(str(configs.exp.seeds),str(gentime))   
    
#     print('savepath: {}'.format(savepath))           
#     return savepath




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
                        
                        
                        
    ## very important using Attridct                         
    configs = AttrDict(configs)
    return configs


