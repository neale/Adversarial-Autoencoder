import numpy as np
import torch
import scipy.misc
import torch.autograd as autograd


def calc_gradient_penalty(args, netD, data, fake):
    alpha = torch.rand(args.batch_size, 1, requires_grad=True)
    if args.dataset in ['mnist', 'fmnist']:
        alpha = alpha.expand(data.size()).cuda()
    else:
        alpha = alpha.expand(args.batch_size, data.nelement()//args.batch_size)
        if args.dataset in ['cifar', 'cifar_hidden']:
            alpha = alpha.contiguous().view(args.batch_size, 3, 32, 32).cuda()
        if args.dataset == 'celeba':
            alpha = alpha.contiguous().view(args.batch_size, 3, 64, 64).cuda()

    interpolates = alpha * data + ((1 - alpha) * fake)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),      
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    if args.dataset != 'mnist':
        gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty

def grad_penalty_3dim(args, netD, data, fake):
    alpha = torch.randn(args.batch_size, 1, requires_grad=True).cuda()
    alpha = alpha.expand(args.batch_size, data.nelement()//args.batch_size)
    alpha = alpha.contiguous().view(args.batch_size, 3, 32, 32)
    interpolates = alpha * data + ((1 - alpha) * fake).cuda()
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


def grad_penalty_1dim(args, netD, data, fake):
    alpha = torch.randn(args.batch_size, 1, requires_grad=True).cuda()
    alpha = alpha.expand(data.size()).cuda()
    interpolates = alpha * data + ((1 - alpha) * fake).cuda()
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


