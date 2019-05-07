import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def create_if_empty(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_ae_image(args, iter, netE, netG, data, save_path):
    encoding = netE(data)
    samples = netG(encoding)
    if args.dataset in ['mnist', 'fmnist']:
        samples = samples.view(args.batch_size, 1, 28, 28)
    else:
        samples = samples.view(-1, *args.shape)
        samples = samples.mul(0.5).add(0.5)
    if args.exp is not '':
        create_if_empty(save_path+args.exp)
        save_path = save_path + args.exp
    save_image(samples, save_path+'/ae_samples_{}.png'.format(iter), normalize=True)


def generate_image(args, iter, netG, save_path):
    if args.dataset in ['mnist', 'fmnist']:
        fixed_noise = torch.randn(args.batch_size, args.z).cuda()
    else:
        fixed_noise = torch.randn(args.batch_size, args.z).cuda()
    samples = netG(fixed_noise)
    if args.dataset in ['mnist', 'fmnist']:
        samples = samples.view(args.batch_size, 1, 28, 28)
    else:
        samples = samples.view(-1, *args.shape)
        samples = samples.mul(0.5).add(0.5)
    if args.exp is not '':
        create_if_empty(save_path+args.exp)
        save_path = save_path + args.exp
    save_image(samples, save_path+'/samples_{}.png'.format(iter), normalize=True)
