import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
#from scipy.misc import imread
from imageio import imread
from torch import Tensor
from torchvision import datasets, transforms
from attrdict import AttrDict
from torch.utils.data import Subset

import PIL
import torch
import torchvision
import random



from collections import OrderedDict


              
corruptions = [ 'shot_noise',
                'speckle_noise',
                'impulse_noise',
                'defocus_blur',
                'gaussian_blur',
                'motion_blur',
                'zoom_blur',
                'snow',
                'fog',
                'brightness',
                'contrast',
                'elastic_transform',
                'pixelate',
                'jpeg_compression',
                'spatter',
                'saturate',
                'frost',    
]
        
        
"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.
Set root to point to the Train/Test folders.
"""


    
## ---------------------------
# main sets
## ---------------------------          
num_workers=4      

abs_path = '~/'      
    



def setup_dataset(configs:AttrDict, fortrain=True):
    trainloader,valloader,testloader = _setup_mainset(configs)
    contextloader = _setup_contextset(configs,fortrain=fortrain)
    oodloader     = _setup_oodset(configs)    
    return trainloader,valloader,testloader,contextloader,oodloader    



def _setup_mainset(configs: AttrDict):    
    batch_size = configs.train.nbatch


        
    if configs.dataset.trainset == 'cifar10':
        #mu,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(  mu, std  )])
  
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std  = [0.229, 0.224, 0.225] 
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        dataset = datasets.CIFAR10(abs_path + '/Datasets/CIFAR10/', download=True, train=True, transform=transform_train)
        testset = datasets.CIFAR10(abs_path + '/Datasets/CIFAR10/', download=True, train=False, transform=transform_test)
        
        n_classes = 10

        

        
    if configs.dataset.trainset == 'cifar100':
        
        #dataset_mean, dataset_std = [0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]

        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]         
        


        #dataset_split = [45000,5000]                                                           
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR100(abs_path + '/Datasets/CIFAR100/', download=True, train=True, transform=transform_train)
        testset = datasets.CIFAR100(abs_path + '/Datasets/CIFAR100/', download=True, train=False, transform=transform_test)
        
        n_classes = 100
        
        
        

    if configs.dataset.trainset == 'tinyimagenet':
        DATA_DIR  = abs_path +'Datasets/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        #VALID_DIR = os.path.join(DATA_DIR, 'val')
        TEST_DIR = os.path.join(DATA_DIR, 'test') 
        
        
        image_size = 32
        image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        
        train_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        test_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        dataset = datasets.ImageFolder(root = TRAIN_DIR,transform=train_transform)
        testset = datasets.ImageFolder(root = TEST_DIR,transform=test_transform)
        
        n_classes = 200


        
        
    ## split train, val, and test set
    
    if configs.dataset.trainset in ['cifar10','cifar100','tinyimagenet']:
        if  configs.dataset.ntrainset >= 45000:
            ntrainset = 45000
            nvalset   =  5000
            
            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
            
            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=train_sampler,  num_workers=num_workers )  ## working in "after periods"
            valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )


        else:
            ntrainset = int(configs.dataset.ntrainset)
            nvalset   = int(0.1*ntrainset)        


            # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)


            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )
            valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )
                



    if configs.dataset.trainset in ['tinyimagenet64s']:

        DATA_DIR  = abs_path +'Datasets/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')  ## label
        #TEST_DIR = os.path.join(DATA_DIR, 'test') ##no label
        
        
        image_size           = 64
        image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] ##v930, nb 256


        n_classes = 200
        train_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        test_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        dataset = datasets.ImageFolder(root = TRAIN_DIR,transform=train_transform)


        if  configs.dataset.ntrainset >= 90000:
            ntrainset = 90000
            nvalset   = 10000 


        else:
            ntrainset = int(configs.dataset.ntrainset)
            nvalset   = int(0.1*ntrainset)        
        

        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)

        testset = datasets.ImageFolder(root = VALID_DIR,transform=train_transform)
        trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )
        valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
        testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )
            
            

    print('traininig: ntrain={}, nval={}, ntest={}'.format(ntrainset,nvalset, len(testloader.dataset)))      


    return trainloader,valloader,testloader







#---------------------------
# context sets
#---------------------------                
def _setup_contextset(configs:AttrDict , fortrain=True):
    batch_size = configs.train.nbatch
    

        
    ## added
    if configs.dataset.contextset == 'cifar10':

        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std  = [0.229, 0.224, 0.225]                                              
        dataset_split = [45000,5000]                                                    
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR10(abs_path + '/Datasets/CIFAR10/', download=True, train=True, transform=transform_train)
        n_classes = 10 
        
    if configs.dataset.contextset == 'cifar100':
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]         
        dataset_split = [45000,5000]                                                           
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR100( abs_path + '/Datasets/CIFAR100/', download=True, train=True, transform=transform_train)
        n_classes = 100

        

    if configs.dataset.contextset == 'tinyimagenet':
        DATA_DIR  = abs_path +'Datasets/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        
        image_size = 32
        image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        ntrainset = 45000
        
        train_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        test_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        dataset = datasets.ImageFolder(root = TRAIN_DIR,transform=train_transform)
        n_classes = 200



    if configs.dataset.trainset in ['cifar10','cifar100','tinyimagenet']:

        ## split train, val, and test set

        if  configs.dataset.ntrainset >= 45000:
            ntrainset = 45000
            nvalset   =  5000 


        else:
            ntrainset = int(configs.dataset.ntrainset)
            nvalset   = int(0.1*ntrainset)        
        


        train_sampler,_ = _random_train_val_sampler(dataset, n_classes, ntrainset, nvalset)



        trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=train_sampler, num_workers=num_workers )




    if configs.dataset.trainset in ['tinyimagenet64s']:

        DATA_DIR  = abs_path +'Datasets/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')  ## label
        #TEST_DIR = os.path.join(DATA_DIR, 'test') ##no label
        
        
        image_size = 64
        image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        n_classes = 200

        
        train_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        test_transform = transforms.Compose([
                        transforms.Resize(image_size), # Resize images to 256 x 256
                        transforms.CenterCrop(image_size), # Center crop image
                        transforms.ToTensor(),  # Converting cropped images to tensors
                        transforms.Normalize(image_mean,image_std) # 
        ])


        dataset = datasets.ImageFolder(root = TRAIN_DIR,transform=train_transform)



        if  configs.dataset.ntrainset >= 90000:
            ntrainset = 90000
            nvalset   = 10000 


        else:
            ntrainset = int(configs.dataset.ntrainset)
            nvalset   = int(0.1*ntrainset)     
            
            
        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
        trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=train_sampler, num_workers=num_workers )


    print('context: nset={}'.format(ntrainset))      

    return trainloader               
        

    
    
    


def _setup_oodset(configs:AttrDict):
    batch_size = configs.train.nbatch        
    if 'oodset' not in configs.dataset.keys():
        return None

    #if isinstance(configs.dataset.oodset, list):
               

    current_ood = configs.dataset.oodset

    if current_ood == 'mnist':        
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        dataset = datasets.MNIST(abs_path +'/Datasets/MNIST/', download=True, train=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.MNIST(abs_path +'/Datasets/MNIST/', download=True, train=False, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=False , num_workers=num_workers)

        ntrainset = 10000
        





    ## uniform ood set 
    if current_ood == 'svhn':        
        dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
        dataset_split = [65931, 7326]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        
        dataset   = datasets.SVHN(abs_path + '/Datasets/SVHN/', download=True, split="test", transform=transform)
              
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=False , num_workers=num_workers)


        ntrainset = len(testloader.dataset)



    if current_ood == 'svhn64':        
        image_size=64
        dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
        dataset_split = [65931, 7326]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(image_size), # Resize images to 256 x 256
                                        transforms.CenterCrop(image_size), # Center crop image
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        
        
        dataset   = datasets.SVHN(abs_path + '/Datasets/SVHN/', download=True, split="train", transform=transform)        
        n_classes = 10
        ntrainset = 10000
        nvalset   = int(0.1*ntrainset)        


        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
        testloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )





    if current_ood == 'lsun32':
        
        dataset_mean, dataset_std = [0.5076, 0.4702, 0.4350], [0.2745, 0.2760, 0.2896]    #lsun32 resize    
                
                
        DATA_PATH = '~/Datasets'
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        # transform_test=transforms.Compose([transforms.ToTensor(),])        

        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=transform_test)

        testloader = torch.utils.data.DataLoader(test_set,  batch_size=batch_size    , shuffle=False , num_workers=num_workers)


    #print('ood set: nset={}'.format(ntrainset))      

            
    return testloader  
    
    
    
    
    







import copy
from torch.utils.data.sampler import SubsetRandomSampler

def _random_train_val_sampler(train_dataset, n_classes, ntrainset=90, nvalset=10):
 
    ntr_samples_per_class = ntrainset // n_classes       
    nval_samples_per_class = nvalset // n_classes
    
    if hasattr(train_dataset,'targets'):
       labels = train_dataset.targets

    if hasattr(train_dataset,'labels'):
       labels = train_dataset.labels


    train_idx, valid_idx = [], []
    for i in range(n_classes):

        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, nval_samples_per_class, replace=False)
        train_samples = np.setdiff1d(c_idx, valid_samples)
        
        
        
        if len(train_samples) >= ntr_samples_per_class:
            train_samples = train_samples[:ntr_samples_per_class]
        
        
        train_idx.append(train_samples)
        valid_idx.append(valid_samples)
        

    train_idx = np.concatenate(train_idx,0) 
    valid_idx = np.concatenate(valid_idx,0)    
    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)

    
    train_sampler = SubsetRandomSampler([*train_idx])
    valid_sampler = SubsetRandomSampler([*valid_idx])
    
    return train_sampler,valid_sampler








