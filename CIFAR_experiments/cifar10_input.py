'''build CIFAR10 dataset'''
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import PIL

def make_dataset(args):
    '''
    augment:standard,rot-default,combi-rot,combi-rot-no-crop
    '''
    if args.augment == 'standard':
        transform_train = transforms.Compose([
                 transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_valid = transforms.Compose([
                 transforms.ToTensor(), 
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
    elif args.augment == 'rot-default':
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_valid = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                
    elif args.augment == 'combi-rot':
        transform_train = transforms.Compose([transforms.RandomApply([transforms.RandomRotation(180)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(180,interpolation=transforms.InterpolationMode.BILINEAR)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(180,interpolation=transforms.InterpolationMode.BICUBIC)], p=0.25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),   
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_valid = transforms.Compose([transforms.RandomApply([transforms.RandomRotation(180)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(180,interpolation=transforms.InterpolationMode.BILINEAR)], p=0.25),
                transforms.RandomApply([transforms.RandomRotation(180,interpolation=transforms.InterpolationMode.BICUBIC)], p=0.25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),   
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                              
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_valid)
    
    ### Split training data  into train, validation sets
    valid_size=0.2  #20% of training set as validation
    shuffle = True
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(1)#random  seed=1
        np.random.shuffle(indices)
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True)  
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=args.val_batch_size, sampler=valid_sampler,
        num_workers=2, pin_memory=True)
        
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
           
    return trainloader,validloader
