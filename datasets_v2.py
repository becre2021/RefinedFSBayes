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


def make_randommonochrome_xc(x):
    x_f = x.flatten(start_dim=2)
    nb,nch,nsiz = x_f.shape

    if nch == 1:
        power_list = []
        for i_x_f in x_f:
            idx=torch.randperm(nsiz)[:1]
            tmp = torch.tensor([i_x_f[0,idx[0]]])[None,:,None,None]
            #print(tmp.shape)
            power_list.append(tmp)
        power_list = torch.cat(power_list,dim=0)


    if nch == 3:
        power_list = []
        for i_x_f in x_f:
            idx=torch.randperm(nsiz)[:3]
            tmp = torch.tensor([i_x_f[0,idx[0]],i_x_f[1,idx[1]],i_x_f[2,idx[2]]])[None,:,None,None]
            #print(tmp.shape)
            power_list.append(tmp)
        power_list = torch.cat(power_list,dim=0)
    
    return x*power_list.to(x.device)



def make_randommonochrome2_xc(x):
    x_f = x.flatten(start_dim=2)
    nb,nch,nsiz = x_f.shape

    if nch == 1:
        power_list = []
        for i_x_f in x_f:
            idx=torch.randperm(nsiz)[:1]
            tmp = torch.tensor([i_x_f[0,idx[0]]])[None,:,None,None]
            #print(tmp.shape)
            power_list.append(tmp)
        power_list = torch.cat(power_list,dim=0)
        
        x_t=transforms.Grayscale(1)(x)


    if nch == 3:
        power_list = []
        for i_x_f in x_f:
            idx=torch.randperm(nsiz)[:3]
            tmp = torch.tensor([i_x_f[0,idx[0]],i_x_f[1,idx[1]],i_x_f[2,idx[2]]])[None,:,None,None]
            #print(tmp.shape)
            power_list.append(tmp)
        power_list = torch.cat(power_list,dim=0)

        x_t=transforms.Grayscale(3)(x)
    
    return x_t*power_list.to(x.device)


def make_rgb_xc(x):
    nb,nch,_,_ = x.shape
    #xc = build_random_monochrome_xc(x)
    random_scale= 2*(torch.rand(nb,3,1,1) - 0.5)
    xc = random_scale.to(x.device)*x
    return xc


#'gaussian_noise',
              
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

# def setup_dataset(configs:AttrDict):
#     trainloader,valloader,testloader = _setup_mainset(configs)
#     contextloader = _setup_contextset(configs)
#     oodloader = _setup_oodset(configs)    
#     return trainloader,valloader,testloader,contextloader,oodloader
    
def setup_dataset(configs:AttrDict, fortrain=True):
    trainloader,valloader,testloader = _setup_mainset(configs)
    contextloader = _setup_contextset(configs,fortrain=fortrain)
    oodloader     = _setup_oodset(configs)    
    return trainloader,valloader,testloader,contextloader,oodloader    
    
#---------------------------
# main sets
#---------------------------                
def _setup_mainset(configs: AttrDict ):    
    batch_size = configs.train.nbatch
    #train_ratio = configs.train.ratio

    if configs.dataset.trainset == 'fmnist':
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        dataset = datasets.FashionMNIST('~/Datasets/FMNIST/', download=True, train=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.FashionMNIST('~/Datasets/FMNIST/', download=True, train=False, transform=transform)
        
    ## reference: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py    
    if configs.dataset.trainset == 'cifar10':
        #mu,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(  mu, std  )])
                                         
                                             
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 
        dataset_mean, dataset_std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]         
        dataset_split = [45000,5000]                                                    
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR10('~/Datasets/CIFAR10/', download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR10('~/Datasets/CIFAR10/', download=True, train=False, transform=transform_test)
        

    ## reference: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py    
    if configs.dataset.trainset == 'cifar100':
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        #dataset_mean, dataset_std = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        dataset_mean, dataset_std = [0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]
        
        dataset_split = [45000,5000]                                                           
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        dataset = datasets.CIFAR100('~/Datasets/CIFAR100/', download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR100('~/Datasets/CIFAR100/', download=True, train=False, transform=transform_test)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=True)    
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=False)    

    #return trainset, valset, testset, dataset    
    return trainloader,valloader,testloader



#---------------------------
# context sets
#---------------------------                
def _setup_contextset(configs:AttrDict , fortrain=True):
    batch_size = configs.train.nbatch

    if configs.dataset.contextset == 'monochrome':
        # return         
        trainloader, valloader, testloader = _setup_mainset(configs)
        if fortrain:
            return trainloader
        else:
            return testloader        
        
    
    if configs.dataset.contextset == 'kmnist':        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5),  )
        ])
        contextdataset = datasets.MNIST('~/Datasets/KMNIST/', download=True, train=True, transform=transform)
        contextloader = torch.utils.data.DataLoader(contextdataset, batch_size=batch_size , shuffle=True)
        
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        dataset = datasets.MNIST('~/Datasets/KMNIST/', download=True, train=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.MNIST('~/Datasets/KMNIST/', download=True, train=False, transform=transform)
        
        
    ## added
    if configs.dataset.contextset == 'cifar10':
        #mu,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(  mu, std  )])
                                         
                                             
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 
        dataset_mean, dataset_std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]         
        dataset_split = [45000,5000]                                                    
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR10('~/Datasets/CIFAR10/', download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR10('~/Datasets/CIFAR10/', download=True, train=False, transform=transform_test)
        
                
        
    if configs.dataset.contextset == 'cifar100':
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]        
        dataset_mean, dataset_std = [0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]
        dataset_split = [45000,5000]                                                           
        transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        dataset = datasets.CIFAR100('~/Datasets/CIFAR100/', download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR100('~/Datasets/CIFAR100/', download=True, train=False, transform=transform_test)
                


    if configs.dataset.contextset == 'tinyimagenet':
        # Define training and validation data paths
        DATA_DIR  = '/home/becre1776/Datasets/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')
        TEST_DIR  = os.path.join(DATA_DIR, 'test')
        
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
        dataset_split = [ntrainset,len(dataset) - ntrainset]
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.ImageFolder(root = TEST_DIR,transform=test_transform)
        

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True)
    #valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=False)    


    if fortrain:
        return trainloader
    else:
        return testloader
    
#---------------------------
# ood sets
#---------------------------        
#---------------------------
# # ood sets
# #---------------------------        
# def _setup_oodset(configs:AttrDict):
#     batch_size = configs.train.nbatch        
#     if 'oodset' not in configs.dataset.keys():
#         return None
     
#     assert type(configs.dataset.oodset) == list
         
#     if configs.dataset.oodset == 'mnist':        
#         dataset_mean = [0.5, ]
#         dataset_std = [0.5, ] 
#         dataset_split = [54000,6000]                                                            
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=dataset_mean, std=dataset_std)
#         ])
#         dataset = datasets.MNIST('~/Datasets/MNIST/', download=True, train=True, transform=transform)
#         trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
#         testset = datasets.MNIST('~/Datasets/MNIST/', download=True, train=False, transform=transform)

#     if configs.dataset.oodset == 'notmnist':
#         dataset_mean = [0.5, ]
#         dataset_std = [0.5, ] 
#         dataset_split = [54000,6000]                                                            
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=dataset_mean, std=dataset_std)
#         ])        
#         dataset = notMNIST(train=True, transform=transform)
#         trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
#         testset = datasets.MNIST(train=False, transform=transform)
                
#     if configs.dataset.oodset == 'svhn':        
#         dataset_mean = [0.4376821, 0.4437697, 0.47280442]
#         dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
#         dataset_split = [65931, 7326]                                                            
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=dataset_mean, std=dataset_std)
#         ])        
#         dataset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="train", transform=transform)
#         trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
#         testset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="test", transform=transform)

#     # if configs.dataset.oodset == 'cifar10c':        
#     #     dataset_mean [0.4376821, 0.4437697, 0.47280442]
#     #     dataset_std [0.19803012, 0.20101562, 0.19703614]        
#     #     dataset_split = [65931, 7326]                                                            
#     #     transform = transforms.Compose([transforms.ToTensor(),
#     #                                     transforms.Normalize(mean=dataset_mean, std=dataset_std)
#     #     ])        
#     #     dataset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="train", transform=transform)
#     #     trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
#     #     testset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="test", transform=transform)

 
#     #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True)
#     #valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=True)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=True)        
    
#     #return trainset, valset, testset, dataset    
#     return testloader



def _setup_oodset(configs:AttrDict):
    batch_size = configs.train.nbatch        
    if 'oodset' not in configs.dataset.keys():
        return None
     
    #assert type(configs.dataset.oodset) == list

    oodset_list = []
    for current_ood in configs.dataset.oodset:         
        if current_ood == 'mnist':        
            dataset_mean = [0.5, ]
            dataset_std = [0.5, ] 
            dataset_split = [54000,6000]                                                            
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])
            dataset = datasets.MNIST('~/Datasets/MNIST/', download=True, train=True, transform=transform)
            trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
            testset = datasets.MNIST('~/Datasets/MNIST/', download=True, train=False, transform=transform)

        if current_ood == 'notmnist':
            dataset_mean = [0.5, ]
            dataset_std = [0.5, ] 
            dataset_split = [54000,6000]                                                            
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])        
            # dataset = notMNIST(train=True, transform=transform)
            # trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
            # testset = datasets.MNIST(train=False, transform=transform)
            
            testset = notMNIST(train=True, transform=transform)
            
                    
        if current_ood == 'svhn':       
            ntest = 10000                
            dataset_mean = [0.4376821, 0.4437697, 0.47280442]
            dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
            dataset_split = [65931, 7326]                                                            
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])        
            dataset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="train", transform=transform)
            trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
            testset = datasets.SVHN('~/Datasets/SVHN/', download=True, split="test", transform=transform)


        if current_ood == 'cifar10c':        
            ntest = 10000   
            i_ntest = ntest // len(corruptions)
            dataset_mean = [0.49139968, 0.48215841, 0.44653091]
            dataset_std  = [0.24703223, 0.24348513, 0.26158784]

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(dataset_mean , dataset_std)
            ])

            
            subsetdataset_list =[]
            for i_corruptions in corruptions:    
                # i_dataset = CIFAR10C(root='~/Datasets/CIFAR10C/',
                #                     name = i_corruptions,
                #                     transform=transform,
                #                     )   
                root = '/home/becre1776/Datasets/CIFAR10C/' 
                i_dataset = CIFAR10C(root = root,
                                    name = i_corruptions,
                                    transform=transform,
                                    )    
                
                #print(i_corruptions, len(i_dataset), i_ntest )
                subsetdataset_list.append( extract_subset(dataset = i_dataset, num_subset = i_ntest, random_subset = True) )
                
            testset = torch.utils.data.ConcatDataset(subsetdataset_list)            

            #print('cifar10c loaded as expected')
    
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True)
        #valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=True)        
        #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=False)        
        oodset_list.append(testloader)
    #return trainset, valset, testset, dataset    
    return oodset_list

    
    
    
path_notmnist_train = '/home/becre1776/Datasets/NOTMNIST/Train/' 
path_notmnist_test = '/home/becre1776/Datasets/NOTMNIST/Test/' 
class notMNIST(Dataset):
    # The init method is called when this class will be instantiated.
    def __init__(self,train=True, transform=None,ntarget=10000):
        if train:
            root = path_notmnist_train       
        else:
            root = path_notmnist_test
        
        Images, Y = [], []
        folders = os.listdir(root)

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Images.append(np.array(imread(img_path)))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, ims))

        # self.data = [(x, y) for x, y in zip(Images, Y)]
        # self.transform = transform
        if train:
            data = [(x, y) for x, y in zip(Images, Y)]
            idx = np.random.randint(0,len(data),len(data))[:ntarget]
            self.data = [data[j] for j in idx]            
            #print(len(self.data))
        else:
            self.data = [(x, y) for x, y in zip(Images, Y)]
                        
        self.transform = transform
        
        
        # The number of items in the dataset
    def __len__(self):
        # return len(self.data)
        return len(self.data)
    
    def __getitem__(self, index):
        # 8 bit images. Scale between [0,1]. This helps speed up our training        
        img = self.data[index][0]/ 255.0
        
        if self.transform is not None:
            img = self.transform(img)        

        # Input for Conv2D should be Channels x Height x Width
        img_tensor = img.view(1,28, 28).float()
        label = self.data[index][1]
        return (img_tensor, label)
    
    
    
    
## CIFAR10 C
class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, transform=None, target_transform=None):        
        assert name in corruptions
        super(CIFAR10C, self).__init__(root, transform=transform,target_transform=target_transform)
        #print('root: {}, name: {}'.format(root, name))
        # data_path = os.path.join(root, name + '.npy')
        # target_path = os.path.join(root, 'labels.npy' )
        data_path   =  root +  name    + '.npy'
        target_path =  root + 'labels' + '.npy'
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)












