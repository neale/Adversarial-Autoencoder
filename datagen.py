import os
import numpy as np
import utils
import torch
import torchvision
import pickle
from torchvision import datasets, transforms
from scipy.misc import imread,imresize
from sklearn.model_selection import train_test_split
from glob import glob


def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data/mnist'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_fashion_mnist(args):
    path = 'data/fashion_mnist'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_cifar(args):
    path = 'data/cifar'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=True, **kwargs)
    return trainloader, testloader


def load_cifar_hidden(args, c_idx):
    path = 'data/cifar'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    def get_classes(target, labels):
        label_indices = []
        for i in range(len(target)):
            if target[i][1] in labels:
                label_indices.append(i)
        return label_indices

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, c_idx))
    trainloader = torch.utils.data.DataLoader(train_hidden, batch_size=args.batch_size,
            shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_test)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, c_idx))
    testloader = torch.utils.data.DataLoader(test_hidden, batch_size=100,
            shuffle=True, **kwargs)
    return trainloader, testloader
