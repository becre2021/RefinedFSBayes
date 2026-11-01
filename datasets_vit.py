import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
#from scipy.misc import imread
from imageio import imread
from torch import Tensor
from torchvision import datasets, transforms


from torch.utils.data import Subset
import PIL
import torch
import torchvision
import random


#from attrdict import AttrDict
from addict import Dict as AttrDict





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
    
num_workers=4      
#num_workers=8      
DATASET_ROOT = '/home/yohan/project_research/datasets'


    
def setup_dataset(configs:AttrDict, fortrain=True):
    if fortrain :  
        trainloader,valloader,testloader = _setup_mainset(configs)
        contextloader                    = _setup_contextset(configs,fortrain=fortrain)
        oodloader                        = _setup_oodset(configs) 
        return trainloader,valloader,testloader,contextloader,oodloader    
    else:
        trainloader,valloader,testloader = _setup_mainset(configs)
        oodloader                        = _setup_oodset(configs)         
        return testloader,oodloader    





#---------------------------
# main sets
#---------------------------                
def _setup_mainset(configs: AttrDict ):    
    
    batch_size = configs.train.nbatch
    batch_size_val = configs.train.nvalbatch            

    

    ##https://github.com/BEPCTA/TransferLearningPyTorch/blob/master/TransferLearningResnet50_Part_1.ipynb
    ##https://github.com/dhruvbird/ml-notebooks/blob/main/Flowers-102-transfer-learning/flowers102-classification-using-pre-trained-models.ipynb
    ##https://github.com/omerb01/TransBoost/blob/main/transboost/transboost.py
    ## https://arxiv.org/pdf/2205.13331.pdf
    
    
    # flowers: (1020, 102),(1020, 102),(6149, 102)

    # dtd: (1880, 47),(1880, 47),(1880, 47)

    # aircraft:(3334, 100), (3333, 100), (3333, 100)

    # pets37: (3282, 37), (333, 37), (3669, 37),

    # caltech101: ( 6943, 101), (867, 101), (867, 101),


    
    if configs.dataset.trainset in ['flowers102','dtd','aircraft','pets37','caltech101']:
        resolution = configs.dataset.resolution if 'resolution' in configs.dataset else 32  

        # Image transformations
        image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=resolution),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            # Validation does not use augmentation
            'val':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # Test does not use augmentation
            'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }



        dataset_path = DATASET_ROOT


        if configs.dataset.trainset == 'flowers102':
            trainset  = datasets.Flowers102(root=dataset_path , split='train', transform=image_transforms['train'], download=True)
            valset    = datasets.Flowers102(root=dataset_path , split='val',   transform=image_transforms['val'], download=True)
            testset   = datasets.Flowers102(root=dataset_path , split='test',  transform=image_transforms['test'], download=True)

            # trainset  = datasets.Flowers102(root=dataset_path + 'flowers-102/', split='train', transform=image_transforms['train'], download=True)
            # valset    = datasets.Flowers102(root=dataset_path + 'flowers-102/', split='val',   transform=image_transforms['val'], download=True)
            # testset   = datasets.Flowers102(root=dataset_path + 'flowers-102/', split='test',  transform=image_transforms['test'], download=True)





        if configs.dataset.trainset == 'pets37':
            dataset  = datasets.OxfordIIITPet(root=dataset_path , split='trainval', transform=image_transforms['train'] , download=True)
            testset   = datasets.OxfordIIITPet(root=dataset_path , split='test',  transform=image_transforms['test']    , download=True)
            
            #breakpoint()

            ## trainset = 3680

            n_classes = 37
            ntrainvalset = 3680
            ntrainset = int(0.9*ntrainvalset )
            nvalset   = ntrainvalset - ntrainset

            configs.dataset.ntrainset= ntrainset

            # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)



            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size       , sampler=train_sampler,  num_workers=num_workers )
            valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size_val   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_val   , shuffle=True  , num_workers=num_workers )
                
                
                
            return trainloader,valloader,testloader



        if configs.dataset.trainset == 'dtd':
            # Preserve the reference's combined train/validation pool.
            train_part = datasets.DTD(root=dataset_path, split='train',
                                      transform=image_transforms['train'], download=True)
            val_part = datasets.DTD(root=dataset_path, split='val',
                                    transform=image_transforms['train'], download=True)
            dataset = torch.utils.data.ConcatDataset([train_part, val_part])
            dataset.targets = list(train_part._labels) + list(val_part._labels)
            testset = datasets.DTD(root=dataset_path, split='test',
                                   transform=image_transforms['test'], download=True)

            ## trainset = 3760

            n_classes = 47
            ntrainvalset = len(dataset)
            ntrainset = int(0.9*ntrainvalset )
            nvalset   = ntrainvalset - ntrainset
            
            configs.dataset.ntrainset=ntrainset
            


            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size       , sampler=train_sampler,  num_workers=num_workers )
            valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size_val   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_val   , shuffle=True  , num_workers=num_workers )
                
                
            return trainloader,valloader,testloader



        if configs.dataset.trainset == 'aircraft':
            # trainset  = datasets.FGVCAircraft(root=dataset_path , split='train', transform=image_transforms['train'], download=False)
            # valset    = datasets.FGVCAircraft(root=dataset_path , split='val',   transform=image_transforms['val'], download=False)
            # testset   = datasets.FGVCAircraft(root=dataset_path , split='test',  transform=image_transforms['test'], download=False)


 
            dataset  =  datasets.FGVCAircraft(root=dataset_path , split='trainval', transform=image_transforms['train'], download=True)
            testset   = datasets.FGVCAircraft(root=dataset_path , split='test',  transform=image_transforms['test'], download=True)



            ## trainset = 3760

            n_classes = 100
            ntrainvalset = len(dataset)
            ntrainset    = int(0.9*ntrainvalset )
            nvalset      = ntrainvalset - ntrainset

            configs.dataset.ntrainset=ntrainset
        

            train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes,ntrainset, nvalset)
            trainloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size       , sampler=train_sampler,  num_workers=num_workers )
            valloader   = torch.utils.data.DataLoader(dataset,  batch_size=batch_size_val   , sampler=valset_sampler, num_workers=num_workers )
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_val   , shuffle=True  , num_workers=num_workers )
                
                
            return trainloader,valloader,testloader






        if configs.dataset.trainset == 'caltech101':


            trainset, valset, testset = prepare_caltech101_dataloaders()







    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size ,     shuffle=True , num_workers=num_workers, pin_memory=True)
    valloader   = torch.utils.data.DataLoader(valset,   batch_size=batch_size_val  , shuffle=True , num_workers=num_workers, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_val  , shuffle=True , num_workers=num_workers, pin_memory=True)    

    return trainloader,valloader,testloader






import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from torch.utils.data import Dataset

class MinusOneLabel(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y - 1





def prepare_caltech101_dataloaders(
    root=os.path.join(DATASET_ROOT, 'caltech101'),
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    drop_background: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
):
    """
    Prepare Caltech-101 dataloaders from an extracted ImageFolder directory.

    Expected structure:
    root/
        101_ObjectCategories/
        airplanes/
        camera/
        ...

    Returns:
    train_loader, val_loader, test_loader, class_names
    """
    root = os.path.expanduser(root)
    data_dir = os.path.join(root, "101_ObjectCategories")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Could not find '{data_dir}'. Make sure you extracted Caltech-101 so that "
            f"'{root}/101_ObjectCategories' exists."
        )

    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])

    # Base dataset (train transform for now; we'll create separate objects for eval transform)
    ds_train_all = datasets.ImageFolder(data_dir, transform=tf_train)

    # Optionally drop BACKGROUND_Google
    if drop_background and "BACKGROUND_Google" in ds_train_all.classes:
        bg_idx = ds_train_all.class_to_idx["BACKGROUND_Google"]
        keep_indices = [i for i, (_, y) in enumerate(ds_train_all.samples) if y != bg_idx]
        ds_train_all = Subset(ds_train_all, keep_indices)

        # Need class_names for the filtered dataset:
        class_names = [c for c in datasets.ImageFolder(data_dir).classes if c != "BACKGROUND_Google"]
    else:
        class_names = ds_train_all.dataset.classes if isinstance(ds_train_all, Subset) else ds_train_all.classes

    n = len(ds_train_all)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Split sizes invalid. Got n={n}, val_ratio={val_ratio}, test_ratio={test_ratio}.")

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(ds_train_all, [n_train, n_val, n_test], generator=g)

    # For eval transforms, we need datasets with tf_eval.
    # If ds_train_all is a Subset, its underlying dataset is ImageFolder with tf_train.
    # We'll rebuild an eval dataset and map indices appropriately.
    base_eval = datasets.ImageFolder(data_dir, transform=tf_eval)

    # If we dropped background, mirror that filtering for eval dataset too.
    if drop_background and "BACKGROUND_Google" in base_eval.classes:
        bg_idx = base_eval.class_to_idx["BACKGROUND_Google"]
        keep_indices_eval = [i for i, (_, y) in enumerate(base_eval.samples) if y != bg_idx]
        base_eval = Subset(base_eval, keep_indices_eval)

    # Helper: convert random_split Subset indices to indices into the eval dataset
    # random_split gives indices relative to ds_train_all (which might itself be a Subset).
    def subset_on_eval(split_subset):
        # split_subset.indices are indices into ds_train_all
        if isinstance(ds_train_all, Subset):
            # ds_train_all.indices maps to original ImageFolder indices
            orig_indices = [ds_train_all.indices[i] for i in split_subset.indices]
        else:
            orig_indices = split_subset.indices

        # Now apply those original indices to base_eval (which may be Subset too)
        if isinstance(base_eval, Subset):
            # base_eval.indices is the kept indices; need positions within that kept list
            inv = {orig_i: pos for pos, orig_i in enumerate(base_eval.indices)}
            eval_indices = [inv[i] for i in orig_indices]
            return Subset(base_eval, eval_indices)
        else:
            return Subset(base_eval, orig_indices)

    val_ds = subset_on_eval(val_subset)
    test_ds = subset_on_eval(test_subset)



    train_subset = MinusOneLabel(train_subset)
    val_ds       = MinusOneLabel(val_ds)
    test_ds      = MinusOneLabel(test_ds)


    # train_loader = DataLoader(
    #     train_subset, batch_size=batch_size, shuffle=True,
    #     num_workers=num_workers, pin_memory=pin_memory,
    #     persistent_workers=persistent_workers
    # )
    # val_loader = DataLoader(
    #     val_ds, batch_size=batch_size, shuffle=False,
    #     num_workers=num_workers, pin_memory=pin_memory,
    #     persistent_workers=persistent_workers
    # )
    # test_loader = DataLoader(
    #     test_ds, batch_size=batch_size, shuffle=False,
    #     num_workers=num_workers, pin_memory=pin_memory,
    #     persistent_workers=persistent_workers
    # )

    #return train_loader, val_loader, test_loader, class_names
    return train_subset, val_ds, test_ds



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
        contextdataset = datasets.MNIST(os.path.join(DATASET_ROOT, 'KMNIST'), download=True, train=True, transform=transform)
        contextloader = torch.utils.data.DataLoader(contextdataset, batch_size=batch_size , shuffle=True)
        
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        dataset = datasets.MNIST(os.path.join(DATASET_ROOT, 'KMNIST'), download=True, train=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.MNIST(os.path.join(DATASET_ROOT, 'KMNIST'), download=True, train=False, transform=transform)
        
        
    ## added
    if configs.dataset.contextset == 'cifar10':
        #mu,std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(  mu, std  )])
                                         
                                             
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 
        dataset_mean, dataset_std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]         
        dataset_split = [45000,5000]          
        resolution = configs.dataset.resolution if 'resolution' in configs.dataset else 32                   
        transform_train=transforms.Compose([transforms.Resize( (resolution,resolution)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(resolution, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.Resize( (resolution,resolution)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        
        dataset = datasets.CIFAR10(os.path.join(DATASET_ROOT, 'CIFAR10'), download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR10(os.path.join(DATASET_ROOT, 'CIFAR10'), download=True, train=False, transform=transform_test)
        
                
        
    if configs.dataset.contextset == 'cifar100':
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]        
        dataset_mean, dataset_std = [0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]
        dataset_split = [45000,5000]                        
        resolution = configs.dataset.resolution if 'resolution' in configs.dataset else 32                   
        transform_train=transforms.Compose([transforms.Resize( (resolution,resolution)),                                   
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(resolution, 4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        transform_test=transforms.Compose([transforms.Resize( (resolution,resolution)),  
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=dataset_mean, std=dataset_std )
        ])        
        
        
        dataset = datasets.CIFAR100(os.path.join(DATASET_ROOT, 'CIFAR100'), download=True, train=True, transform=transform_train)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.CIFAR100(os.path.join(DATASET_ROOT, 'CIFAR100'), download=True, train=False, transform=transform_test)
                
    
    
    if configs.dataset.contextset == 'tinyimagenet':
        # Define training and validation data paths
        #DATA_DIR  = '/home/becre1776/Datasets/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
        DATA_DIR = os.path.join(DATASET_ROOT, 'tiny-imagenet-200')
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')
        #TEST_DIR  = os.path.join(DATA_DIR, 'test')
        
        #image_size = 32
        image_size = configs.dataset.resolution
        image_mean,image_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        #ntrainset = 45000
        ntrainset = configs.dataset.ntrainset
        
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
        #testset = datasets.ImageFolder(root = TEST_DIR,transform=test_transform)
        


    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True,  num_workers=num_workers, pin_memory=True)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=True,  num_workers=num_workers, pin_memory=True)    

    return trainloader

    # if fortrain:
    #     return trainloader
    # else:
    #     return testloader
    
    
    


def _setup_oodset(configs:AttrDict):
    batch_size = configs.train.nbatch            
    batch_size_val = configs.train.nvalbatch            

    # if 'oodset' not in configs.dataset.keys():
    #     return None
     
    #assert type(configs.dataset.oodset) == list
    resolution = configs.dataset.resolution if 'resolution' in configs.dataset else 32          

    oodset_list = []
    #for current_ood in configs.dataset.oodset:         

    if configs.dataset.oodset == 'mnist':        
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        dataset = datasets.MNIST(os.path.join(DATASET_ROOT, 'MNIST'), download=True, train=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(dataset, dataset_split )
        testset = datasets.MNIST(os.path.join(DATASET_ROOT, 'MNIST'), download=True, train=False, transform=transform)

    if configs.dataset.oodset == 'notmnist':
        dataset_mean = [0.5, ]
        dataset_std = [0.5, ] 
        dataset_split = [54000,6000]                                                            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        
        
        testset = notMNIST(train=True, transform=transform)
        

                
    if configs.dataset.oodset == 'svhn':       
        ntest = 10000                
        dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        dataset_std  = [0.19803012, 0.20101562, 0.19703614]        
        dataset_split = [65931, 7326]                                                            
        transform = transforms.Compose([transforms.Resize( (resolution,resolution)),    
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])        

        #dataset = datasets.SVHN(os.path.join(DATASET_ROOT, 'SVHN'), download=True, split="train", transform=transform)
        dataset = datasets.SVHN(os.path.join(DATASET_ROOT, 'SVHN'), download=True, split="train", transform=transform)

        #trainset, valset = torch.utils.data.random_split(dataset, dataset_split )        
        #testset = datasets.SVHN(os.path.join(DATASET_ROOT, 'SVHN'), download=True, split="test", transform=transform)

        n_classes = 10
        ntrainset = 10000
        nvalset   = int(0.1*ntrainset)        

        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes, ntrainset, nvalset)
        testloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )


        return testloader
    
    
        
    if configs.dataset.oodset == 'cifar100':
        #dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]        
        dataset_mean, dataset_std = [0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]
        dataset_split = [45000,5000]                        
        resolution = configs.dataset.resolution if 'resolution' in configs.dataset else 32                   
        transform = transforms.Compose([transforms.Resize( (resolution,resolution)),    
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])                                
                                                
        #dataset = datasets.CIFAR100(os.path.join(DATASET_ROOT, 'CIFAR100'), download=True, train=True, transform=transform)
        dataset = datasets.CIFAR100(os.path.join(DATASET_ROOT, 'CIFAR100'), download=True, train=True, transform=transform)


        n_classes = 100
        ntrainset = 10000
        nvalset   = int(0.1*ntrainset)        

        # train_sampler,valset_sampler = _random_train_val_sampler(dataset, n_classes, nvalset)
        train_sampler,_ = _random_train_val_sampler(dataset, n_classes, ntrainset, nvalset)
        testloader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size    , sampler=train_sampler,  num_workers=num_workers )
                

        return testloader



    # if configs.dataset.oodset == 'cifar10c':        
    #     ntest = 10000   
    #     i_ntest = ntest // len(corruptions)
    #     dataset_mean = [0.49139968, 0.48215841, 0.44653091]
    #     dataset_std  = [0.24703223, 0.24348513, 0.26158784]

    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize(dataset_mean , dataset_std)
    #     ])

        
    #     subsetdataset_list =[]
    #     for i_corruptions in corruptions:    
    #         # i_dataset = CIFAR10C(root='~/Datasets/CIFAR10C/',
    #         #                     name = i_corruptions,
    #         #                     transform=transform,
    #         #                     )   
    #         root = '/home/becre1776/Datasets/CIFAR10C/' 
    #         i_dataset = CIFAR10C(root = root,
    #                             name = i_corruptions,
    #                             transform=transform,
    #                             )    
            
    #         #print(i_corruptions, len(i_dataset), i_ntest )
    #         subsetdataset_list.append( extract_subset(dataset = i_dataset, num_subset = i_ntest, random_subset = True) )
            
    #     testset = torch.utils.data.ConcatDataset(subsetdataset_list)            

        #print('cifar10c loaded as expected')

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size , shuffle=True)
    #valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size , shuffle=True)
    
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size , shuffle=True )        
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_val , shuffle=True )        

    return testloader

    
    
    
    
    
    
path_notmnist_train = os.path.join(DATASET_ROOT, 'NOTMNIST', 'Train') 
path_notmnist_test = os.path.join(DATASET_ROOT, 'NOTMNIST', 'Test') 
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






import copy
from torch.utils.data.sampler import SubsetRandomSampler

def _random_train_val_sampler(train_dataset, n_classes, ntrainset=90, nvalset=10):
 
    ntr_samples_per_class = ntrainset // n_classes       
    nval_samples_per_class = nvalset // n_classes
    
    if hasattr(train_dataset,'targets'):
       labels = train_dataset.targets

    if hasattr(train_dataset,'labels'):
       labels = train_dataset.labels

    if hasattr(train_dataset,'_labels'):
       labels = train_dataset._labels

    # if hasattr(train_dataset,'classes'):
    #    labels = train_dataset.classes
       
       

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






