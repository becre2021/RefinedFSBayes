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
num_workers=8      
#num_workers=4      

abs_path = os.path.expanduser('~/')      

    



def setup_dataset(configs:AttrDict, fortrain=True):
    trainloader,valloader,testloader = _setup_mainset(configs)
    contextloader = _setup_contextset(configs,fortrain=fortrain)
    oodloader     = _setup_oodset(configs)    
    return trainloader,valloader,testloader,contextloader,oodloader    



def _setup_mainset(configs: AttrDict):    
    batch_size = configs.train.nbatch
        
        
    if configs.dataset.trainset == 'cifar10':
                                             
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
        
        
        #dataset = datasets.CIFAR100(abs_path + '/Datasets/CIFAR100/', download=True, train=True, transform=transform_train)
        #trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        
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


        
    if configs.dataset.trainset in ['cifar10','cifar100','tinyimagenet']:
        valset = copy.copy(dataset)
        valset.transform = transform_test if configs.dataset.trainset != 'tinyimagenet' else test_transform
        if  configs.dataset.ntrainset >= 45000:
            ntrainset = 45000
            nvalset   =  5000
            
            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
          

          
            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=train_sampler,  num_workers=num_workers )  
            valloader   = torch.utils.data.DataLoader(valset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )


        else:
            ntrainset = int(configs.dataset.ntrainset)
            nvalset   = int(0.1*ntrainset)        


            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)

            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )
            valloader   = torch.utils.data.DataLoader(valset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )
                



    if configs.dataset.trainset in ['tinyimagenet64s']:

        DATA_DIR  = abs_path +'Datasets/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')  ## label
        #TEST_DIR = os.path.join(DATA_DIR, 'test') ##no label
        
        
        image_size = 64
        #image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] ##v910
        #image_mean,image_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  ##v920
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
        

        train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)

        valset = copy.copy(dataset)
        valset.transform = test_transform
        testset = datasets.ImageFolder(root = VALID_DIR,transform=test_transform)
        trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )
        valloader   = torch.utils.data.DataLoader(valset,  batch_size=batch_size   , sampler=valset_sampler, num_workers=num_workers )
        testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size   , shuffle=False  , num_workers=num_workers )
            
            

    print('traininig: ntrain={}, nval={}, ntest={}'.format(ntrainset,nvalset, len(testloader.dataset)))      


    return trainloader,valloader,testloader







#---------------------------
# context sets
#---------------------------                
def _setup_contextset(configs:AttrDict , fortrain=True):
    batch_size = configs.train.nbatch
    

    if configs.dataset.contextset == 'cifar10':
                                             
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
            
            
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
        trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size   , sampler=train_sampler, num_workers=num_workers )


    print('context: nset={}'.format(ntrainset))      

    return trainloader               
        

    
    
    


def _setup_oodset(configs:AttrDict):
    batch_size = configs.train.nbatch        
    if 'oodset' not in configs.dataset.keys():
        return None

    current_ood = configs.dataset.oodset
        

    ## uniform ood set 
    if current_ood == 'svhn':        
        dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
        dataset_split = [65931, 7326]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        
        dataset   = datasets.SVHN(abs_path + '/Datasets/SVHN/', download=True, split="train", transform=transform)
        n_classes = 10
        ntrainset = 10000
        nvalset   = int(0.1*ntrainset)  
              


        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
        testloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )





    ## uniform ood set 
    if current_ood == 'svhn2':        
        dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
        dataset_split = [65931, 7326]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        
        dataset   = datasets.SVHN(abs_path + '/Datasets/SVHN/', download=True, split="test", transform=transform)
              
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=False , num_workers=num_workers)


        ntrainset = len(testloader.dataset)



    if current_ood == 'svhn64s':        
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
        
                       
        dataset_mean, dataset_std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  
                
        DATA_PATH = abs_path + '/Datasets/'
        transform_test=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        # transform_test=transforms.Compose([transforms.ToTensor(),])        

        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=transform_test)

        testloader = torch.utils.data.DataLoader(test_set,  batch_size=batch_size    , shuffle=False , num_workers=num_workers)



    if current_ood == 'place365':
    

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ])

        dataset = Places365OOD(
            root=os.environ.get("PLACES365_ROOT", "/home/yohan/Datasets/places365/val_256"),
            transform=transform
        )

        testloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )



    return testloader  
            
    
    
    
    
    
    
    
    
    

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Places365OOD(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not self.images:
            raise ValueError(f"No Places365 images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1  # dummy label
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
path_notmnist_train = abs_path + '/Datasets/NOTMNIST/Train/' 
path_notmnist_test =  abs_path + '/Datasets/NOTMNIST/Test/' 
import random

class notMNIST(Dataset):
    # The init method is called when this class will be instantiated.
    def __init__(self,train=True, transform=None):
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
                    

        data = [(x, y) for x, y in zip(Images, Y)]   
        random.shuffle(data)
        self.data = data[:10000]   
        self.transform = transform
        
        # The number of items in the dataset
    def __len__(self):
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








import copy
from torch.utils.data.sampler import SubsetRandomSampler

def _random_train_val_sampler(train_dataset, n_classes, ntrainset=90, nvalset=10):
 
    for name, value in [('n_classes', n_classes), ('ntrainset', ntrainset), ('nvalset', nvalset)]:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if n_classes == 0:
        raise ValueError("n_classes must be positive")
    ntr_samples_per_class, train_remainder = divmod(ntrainset, n_classes)
    nval_samples_per_class, val_remainder = divmod(nvalset, n_classes)
    train_counts = np.full(n_classes, ntr_samples_per_class, dtype=int)
    valid_counts = np.full(n_classes, nval_samples_per_class, dtype=int)
    train_counts[:train_remainder] += 1
    valid_counts[(np.arange(val_remainder) + train_remainder) % n_classes] += 1
    
    if hasattr(train_dataset,'targets'):
       labels = train_dataset.targets

    elif hasattr(train_dataset,'labels'):
       labels = train_dataset.labels
    elif hasattr(train_dataset,'_labels'):
       labels = train_dataset._labels
    else:
       raise ValueError('Dataset must expose targets, labels, or _labels')


    train_idx, valid_idx = [], []
    for i in range(n_classes):

        c_idx = (np.array(labels) == i).nonzero()[0]
        if len(c_idx) < train_counts[i] + valid_counts[i]:
            raise ValueError(
                f"Class {i} has {len(c_idx)} samples, but the balanced split "
                f"requires {train_counts[i]} training and {valid_counts[i]} validation samples"
            )
        valid_samples = np.random.choice(c_idx, valid_counts[i], replace=False)
        train_samples = np.setdiff1d(c_idx, valid_samples)
        
        
        
        # Preserve reference selection for sizes divisible by n_classes.
        train_samples = train_samples[:train_counts[i]]
        
        
        train_idx.append(train_samples)
        valid_idx.append(valid_samples)
        

    train_idx = np.concatenate(train_idx,0) 
    valid_idx = np.concatenate(valid_idx,0)    
    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)

    
    train_sampler = SubsetRandomSampler([*train_idx])
    valid_sampler = SubsetRandomSampler([*valid_idx])
    
    return train_sampler,valid_sampler


