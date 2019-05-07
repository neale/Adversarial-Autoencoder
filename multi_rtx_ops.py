import numpy as np
import torch
import scipy.misc
import torch.autograd as autograd


def calc_gradient_penalty(args, netD, data, fake):
    alpha = torch.rand(args.batch_size, 1, requires_grad=True)
    alpha = alpha.expand(args.batch_size, data.nelement()//args.batch_size)
    alpha = alpha.contiguous().view(args.batch_size, *args.shape).cuda()
    interpolates = alpha * data + ((1 - alpha) * fake)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),      
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    #if args.:
    #    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty



