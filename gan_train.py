import os
import sys
import time
import argparse
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
from torchvision.utils import save_image
import ops
import plot
import utils
import datagen
import models.encoders as encoders
import models.decoders as decoders
import models.discriminators as discriminators

def load_args():

    parser = argparse.ArgumentParser(description='aae-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space size')
    parser.add_argument('--dim', default=64, type=int, help='network channels')
    parser.add_argument('--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('--lrG', default=1e-4, type=int, help='LR_Enc_Dec')
    parser.add_argument('--lrD', default=1e-4, type=int, help='LR_Disc')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--dataset', default='cifar')
    args = parser.parse_args()
    return args


def load_models(args):
    if args.dataset in ['mnist', 'fmnist']:
        netG = decoders.ToyDecoder(args).cuda()
        netD = discriminators.ToyDiscriminator(args).cuda()

    elif args.dataset in ['cifar', 'cifar_hidden']:
        netG = decoders.DeepDecoder(args).cuda()
        netD = discriminators.DeepDiscriminator(args).cuda()

    elif args.dataset == 'celeba':
        netG = decoders.DeepDecoder(args).cuda()
        netD = discriminators.DeepDiscriminator(args).cuda()

    print (netG, netD)
    return (netG, netD)


def load_data(args):
    if args.dataset == 'mnist':
        return datagen.load_mnist(args)
    elif args.dataset == 'cifar':
        return datagen.load_cifar(args)
    elif args.dataset == 'fmnist':
        return datagen.load_fashion_mnist(args)
    elif args.dataset == 'cifar_hidden':
        class_list = [0] ## just load class 0
        return datagen.load_cifar_hidden(args, class_list)
    else:
        print ('Dataset not specified correctly')
        print ('choose --dataset <mnist, fmnist, cifar, cifar_hidden>')
        sys.exit(0)

def train():
    torch.manual_seed(8734)
    args = load_args()
    utils.create_if_empty('models')
    utils.create_if_empty('models/{}'.format(args.dataset))
    utils.create_if_empty('plots/{}'.format(args.dataset))
    train_gen, test_gen = load_data(args)

    single = next(iter(train_gen))[0]
    args.shape = list(single.size())[1:]
    print ('Data shape: ', args.shape)
    save_dir = 'plots/{}/'.format(args.dataset)
    save_image(single, save_dir+'reals.png', normalize=True)
    
    one = torch.tensor(1.).cuda()
    mone = one * -1
    iteration = 0 

    netG, netD = load_models(args)
    optimG = Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.9), weight_decay=1e-4)
    schedulerG = ExponentialLR(optimG, gamma=0.99) 
    schedulerD = ExponentialLR(optimD, gamma=0.99)

    for epoch in range(args.epochs):
        for i, (data, targets) in enumerate(train_gen):
            netG.zero_grad(); netD.zero_grad()
            """ Update D network """
            for p in netD.parameters():  
                p.requires_grad = True 
            for _ in range(5):
                # train with2real data
                data = data.cuda()
                if args.dataset in ['mnist', 'fmnist']:
                    data = data.view(args.batch_size, 784).cuda()
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
                if args.dataset in ['mnist', 'fmnist']:
                    gp = ops.grad_penalty_1dim(args, netD, data, fake)
                else:
                    gp = ops.grad_penalty_3dim(args, netD, data, fake)
                gp.backward()
                D_cost = D_fake - D_real + gp
                Wasserstein_D = D_real - D_fake
                optimD.step()

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
            optimG.step() 

            schedulerD.step()
            schedulerG.step()
            # Write logs and save samples 
            if iteration % 200 == 0:
                with torch.no_grad():
                    data = data.cuda()
                    utils.generate_image(args, iteration, netG, save_dir)
                print('ITER: ', iteration, 'D cost', D_cost.cpu().item())
                print('ITER: ', iteration,'G cost', G_cost.cpu().item())
                print('ITER: ', iteration,'GP', gp.cpu().item())
                print('ITER: ', iteration,'w1 distance', Wasserstein_D.cpu().item())
            
            if iteration % 5000 == 0:
                utils.save_model(netG, optimG, iteration,
                        'models/{}/G_{}'.format(args.dataset, iteration))
                utils.save_model(netD, optimD, iteration, 
                        'models/{}/D_{}'.format(args.dataset, iteration))
            iteration += 1

        
if __name__ == '__main__':
    train()
