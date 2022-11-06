import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import random
import numpy as np
from dataset import glow_dataset

def data_loader(dir, dataset, model, batch_size, workers):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(root=dir, train=False, download=False, transform=val_transform)
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        train_set = datasets.CIFAR100(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(root=dir, train=False, download=False, transform=val_transform)
    elif dataset == 'imagenet':
        traindir = os.path.join(dir, 'imagenet/train')
        valdir = os.path.join(dir, 'imagenet/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.ImageFolder(traindir, transform=train_transform)
        val_set = datasets.ImageFolder(valdir, transform=val_transform)
    elif dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.MNIST(root=dir, train=False, download=False, transform=val_transform)

    elif dataset == 'glow_cifar10':
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # No normalization applied, since Glow expects inputs in (0, 1)
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        # 先设定，以后修改
        num_class = 0
        if num_class != -1 and num_class != -2:
            train_set = glow_dataset(num_class % 10, transform_train, test=False)
            val_set = glow_dataset(num_class % 10, transform_test, test=True)
        elif num_class == -1:
            train_set = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=True, download=True,
                                                    transform=transform_train)
            val_set = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=False, download=True,
                                                   transform=transform_test)
        else:
            train_set = glow_dataset(num_class % 10, transform_train, test=False, rotation_data=args.rotation_data)
            val_set = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=False, download=True,
                                                   transform=transform_test)
    else:
        assert False, 'No Such Dataset'

    if model == 'glow':
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                                                 pin_memory=True)

    return train_loader, val_loader
