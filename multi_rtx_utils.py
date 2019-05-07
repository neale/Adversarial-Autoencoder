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
    samples = samples.view(args.batch_size, *args.shape)
    if args.nc > 1:
        samples = samples.mul(0.5).add(0.5)
    save_image(samples, save_path+'ae_samples_{}.png'.format(iter))


def generate_image(args, iter, netG, save_path):
    noise = torch.randn(32, args.z).cuda()
    samples = netG(noise)
    samples = samples.view(-1, *args.shape)
    if args.nc > 1:
        samples = samples.mul(0.5).add(0.5)
    save_image(samples, save_path+'samples_{}.png'.format(iter))
