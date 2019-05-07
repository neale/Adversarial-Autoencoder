import os
import sys
import time
import argparse
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
import multi_rtx_ops as ops
import multi_rtx_utils as utils
import datagen
import models.encoders as encoders
import models.decoders as decoders
import models.discriminators as discriminators


def load_args():

    parser = argparse.ArgumentParser(description='aae-wgan')
    parser.add_argument('--z', default=100, type=int, help='latent space size')
    parser.add_argument('--dim', default=64, type=int, help='network channels')
    parser.add_argument('-l', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--dataset', default='celeba')
    parser.add_argument('--use_spectral_norm', default=False)
    args = parser.parse_args()
    return args


def load_models(args):
    print (args.gpus)
    if args.dataset in ['mnist', 'fmnist']:
        """
        netE = nn.DataParallel(encoders.SmallEncoder(args)).cuda()
        netG = nn.DataParallel(decoders.SmallDecoder(args)).cuda()
        netD = nn.DataParallel(discriminators.SmallDiscriminator(args)).cuda()
        """
        netE = encoders.SmallEncoder(args).cuda()
        netG = decoders.SmallDecoder(args).cuda()
        netD = discriminators.SmallDiscriminator(args).cuda()
    elif args.dataset in ['cifar', 'cifar_hidden']:
        netE = nn.DataParallel(encoders.DeepEncoder(args)).cuda()
        netG = nn.DataParallel(decoders.DeepDecoder(args)).cuda()
        netD = nn.DataParallel(discriminators.DeepDiscriminator(args)).cuda()

    elif args.dataset == 'celeba':
        netE = nn.DataParallel(encoders.CELEBAencoder(args)).cuda()
        netG = nn.DataParallel(decoders.CELEBAgenerator(args)).cuda()
        netD = nn.DataParallel(discriminators.CELEBAdiscriminator(args)).cuda()
    
    print (netG, netD, netE)
    return (netG, netD, netE)


def set_data_params(args):
    if args.dataset in ['mnist', 'fmnist']:
        args.shape = (1, 32, 32,)
        args.nc = 1
    elif args.dataset in ['cifar', 'cifar_hidden']:
        args.shape = (3, 32, 32)
        args.nc = 3
    return


def load_data(args):
    if args.dataset == 'mnist':
        data =  datagen.load_mnist(args)
    elif args.dataset == 'cifar':
        data = datagen.load_cifar(args)
    elif args.dataset == 'fmnist':
        data = datagen.load_fashion_mnist(args)
    elif args.dataset == 'cifar_hidden':
        class_list = [0] ## just load class 0
        data = datagen.load_cifar_hidden(args, class_list)
    else:
        print ('Dataset not specified correctly')
        print ('choose --dataset <mnist, fmnist, cifar, cifar_hidden>')
        sys.exit(0)
    set_data_params(args)
    return data


def train():
    args.batch_size *= args.gpus
    print ('training with effective batch size: {}'.format(args.batch_size))
    train_gen, test_gen = load_data(args)
    netG, netD, netE = load_models(args)

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999),
            weight_decay=1e-4)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999),
            weight_decay=1e-4)
    optimizerE = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.999),
            weight_decay=1e-4)

    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99) 
    schedulerE = optim.lr_scheduler.ExponentialLR(optimizerE, gamma=0.99)
    
    utils.create_if_empty('models')
    utils.create_if_empty('models/{}'.format(args.dataset))
    utils.create_if_empty('plots/{}'.format(args.dataset))
    
    single = next(iter(train_gen))[0]
    save_dir = 'plots/{}/'.format(args.dataset)
    save_image(single, save_dir+'reals.png', normalize=True)
    ae_criterion = nn.MSELoss()
    one = torch.tensor(1.).cuda()
    mone = one * -1
    iteration = 0 
    for epoch in range(args.epochs):
        for i, (data, targets) in enumerate(train_gen):
            """ Update AutoEncoder """
            
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            netE.zero_grad()
            data = data.cuda()
            encoding = netE(data)
            fake = netG(encoding)
            ae_loss = ae_criterion(fake, data)
            ae_loss.backward(one)
            optimizerE.step()
            optimizerG.step()
            
            """ Update D network """
            for p in netD.parameters():  
                p.requires_grad = True 
            for _ in range(5):
                # train with real data
                data = data.cuda()
                netD.zero_grad()
                D_real = netD(data).mean()
                D_real.backward(mone, retain_graph=True)
                # train with fake data
                noise = torch.randn(args.batch_size, args.z).cuda()
                noise.requires_grad_(True)
                with torch.no_grad():
                    fake = netG(noise)
                fake.requires_grad_(True)
                D_fake = netD(fake).mean()
                D_fake.backward(one, retain_graph=True)
                # train with gradient penalty 
                gp = ops.calc_gradient_penalty(args, netD, data, fake)
                gp.backward()
                D_cost = D_fake - D_real + gp
                Wasserstein_D = D_real - D_fake
                optimizerD.step()

            # Update generator network (GAN)
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            noise = torch.randn(args.batch_size, args.z).cuda()
            noise.requires_grad_(True)
            fake = netG(noise)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step() 

            schedulerD.step()
            schedulerG.step()
            schedulerE.step()
            # Write logs and save samples 
            if iteration % 200 == 0:
                with torch.no_grad():
                    data = data.cuda()
                    utils.generate_image(args, iteration, netG, save_dir)
                    utils.generate_ae_image(args, iteration, netE, netG, data, save_dir)
                print('ITER: ', iteration, 'D cost', D_cost.cpu().item())
                print('ITER: ', iteration,'G cost', G_cost.cpu().item())
                print('ITER: ', iteration,'w1 distance', Wasserstein_D.cpu().item())
                print('ITER: ', iteration,'ae cost', ae_loss.data.cpu().item())
            
            if iteration % 1000 == 0:
                utils.save_model(netG, optimizerG, iteration,
                        'models/{}/G_{}'.format(args.dataset, iteration))
                utils.save_model(netD, optimizerD, iteration, 
                        'models/{}/D_{}'.format(args.dataset, iteration))
            iteration += 1

        
if __name__ == '__main__':
    args = load_args()
    train()
