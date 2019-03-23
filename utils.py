import torch
import torch.autograd as autograd
import numpy as np
from data import mnist
from data import cifar10
from data import celeba
from scipy.misc import imsave
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image


def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def generate_ae_image(iter, netE, netG, save_path, args, real_data):
    batch_size = args.batch_size
    datashape = netE.shape
    encoding = netE(real_data)
    samples = netG(encoding)
    if netG._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/ae_samples_{}.jpg'.format(iter))


def generate_image(iter, model, save_path, args):
    batch_size = args.batch_size
    datashape = model.shape
    if model._name == 'mnistG':
        fixed_noise_128 = torch.randn(batch_size, args.dim).cuda()
    else:
        fixed_noise_128 = torch.randn(128, args.dim).cuda()
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    if model._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.001)

    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)


