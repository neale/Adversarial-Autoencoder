import torch
import torch.nn as nn
import torch.nn.functional as F

## baseline decoder, ConvBNReLU blocks
class SmallDecoder(nn.Module):
    def __init__(self, args):
        super(SmallDecoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 4*4*4*self.dim)
        self.conv1 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(2*self.dim, self.dim, 2, stride=2)
        self.conv3 = nn.ConvTranspose2d(self.dim, self.nc, 2, stride=2)
        self.bn1 = nn.BatchNorm1d(4*4*4*self.dim)
        self.bn2 = nn.BatchNorm2d(2*self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)

    def forward(self, x):
        x = F.elu(self.bn1(self.linear1(x)))
        x = x.view(-1, 4*self.dim, 4, 4)
        x = F.elu(self.bn2(self.conv1(x)))
        x = F.elu(self.bn3(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        x = x.view(-1, *self.shape)
        return x

## VGG style decoder, deep version of the baseline
## Conv2d(input, output, kernel_size, stride, input padding, output_padding)
## Transpose Conv sometimes requires output padding to be set. see pytorch docs
class DeepDecoder(nn.Module):
    def __init__(self, args):
        super(DeepDecoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 4*4*4*2*self.dim)
        self.conv1 = nn.ConvTranspose2d(8*self.dim, 8*self.dim, 3, 2, 2)
        self.conv2 = nn.ConvTranspose2d(8*self.dim, 8*self.dim, 3, 1, 1)
        self.conv3 = nn.ConvTranspose2d(8*self.dim, 8*self.dim, 3, 2, 2)
        self.conv4 = nn.ConvTranspose2d(8*self.dim, 4*self.dim, 3, 1, 1)
        self.conv5 = nn.ConvTranspose2d(4*self.dim, 4*self.dim, 3, 2, 2, 1)
        self.conv6 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, 3, 1, 2)
        self.conv7 = nn.ConvTranspose2d(2*self.dim, self.dim, 3, 2, 2)
        self.conv8 = nn.ConvTranspose2d(self.dim, 3, 3, 2, 2, 1)
        self.bn1 = nn.BatchNorm1d(4*4*4*2*self.dim)
        self.bn2 = nn.BatchNorm2d(8*self.dim)
        self.bn3 = nn.BatchNorm2d(8*self.dim)
        self.bn4 = nn.BatchNorm2d(8*self.dim)
        self.bn5 = nn.BatchNorm2d(4*self.dim)
        self.bn6 = nn.BatchNorm2d(4*self.dim)
        self.bn7 = nn.BatchNorm2d(2*self.dim)
        self.bn8 = nn.BatchNorm2d(self.dim)

    def forward(self, x):
        x = F.elu(self.bn1(self.linear1(x)))
        x = x.view(-1, 4*2*self.dim, 4, 4)
        x = F.elu(self.bn2(self.conv1(x)))
        x = F.elu(self.bn3(self.conv2(x)))
        x = F.elu(self.bn4(self.conv3(x)))
        x = F.elu(self.bn5(self.conv4(x)))
        x = F.elu(self.bn6(self.conv5(x)))
        x = F.elu(self.bn7(self.conv6(x)))
        x = F.elu(self.bn8(self.conv7(x)))
        x = torch.tanh(self.conv8(x))
        x = x.view(-1, *self.shape)
        return x


## very simple decoder for 1d tasks like MNIST or Gaussians
class ToyDecoder(nn.Module):
    def __init__(self, args):
        super(ToyDecoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 4*4*4*self.dim)
        self.conv1 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5)
        self.conv2 = nn.ConvTranspose2d(2*self.dim, self.dim, 5)
        self.conv3 = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.elu(self.linear1(x))
        x = x.view(-1, 4*self.dim, 4, 4)
        x = self.elu(self.conv1(x))
        x = x[:, :, :7, :7]
        x = self.elu(self.conv2(x))
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = x.view(-1, 28*28)
        return x
