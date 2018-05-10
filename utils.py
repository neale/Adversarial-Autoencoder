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


def get_transform(args):
    if args.dataset == 'celeba':
        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size,
                offset_width:offset_width + crop_size]
        preprocess = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Lambda(crop),
                    transforms.ToPILImage(),
                    transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        return preprocess


def dataset_iterator(args):
    transform = get_transform(args)
    if args.dataset == 'mnist':
        train_gen, dev_gen, test_gen = mnist.load(args.batch_size, args.batch_size)
    if args.dataset == 'cifar10':
        data_dir = '/data0/images/cifar-10-batches-py/'
        train_gen, dev_gen = cifar10.load(args.batch_size, data_dir)
        test_gen = None
    if args.dataset == 'celeba':
        data_dir = '/data0/images/celeba'
        data = datasets.ImageFolder(data_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=4)
        return data_loader
        #train_gen = celeba.load(args.batch_size, data_dir+'/train/')
        #dev_gen = celeba.load(args.batch_size, data_dir+'/test/')
        #test_gen = None

    return (train_gen, dev_gen, test_gen)


def inf_train_gen(train_gen):
    while True:
        for i, (images, _) in enumerate(train_gen):
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


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


