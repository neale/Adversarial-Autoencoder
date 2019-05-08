import torch
import torch.nn as nn
import torch.nn.functional as F

## baseline encoder, just ConvBNReLU stacks
class SmallEncoder(nn.Module):
    def __init__(self, args):
        super(SmallEncoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(self.nc, self.dim, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 3, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim*2)
        self.bn3 = nn.BatchNorm2d(self.dim*4)
        self.linear1 = self.linear = nn.Linear(4*4*4*self.dim, self.z)

    def forward(self, input):
        x = input.view(-1, *self.shape)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        return x

## VGG style. Similar to the baseline encoder, just deeper
class DeepEncoder(nn.Module):
    def __init__(self, args):
        super(DeepEncoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(3, self.dim, 3, 2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, 2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 3, 1, padding=2)
        self.conv4 = nn.Conv2d(4*self.dim, 4*self.dim, 3, 2, padding=2)
        self.conv5 = nn.Conv2d(4*self.dim, 8*self.dim, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(8*self.dim, 8*self.dim, 3, 2, padding=2)
        self.conv7 = nn.Conv2d(8*self.dim, 8*self.dim, 3, 1, padding=1)
        self.conv8 = nn.Conv2d(8*self.dim, 8*self.dim, 3, 2, padding=2)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim*2)
        self.bn3 = nn.BatchNorm2d(self.dim*4)
        self.bn4 = nn.BatchNorm2d(self.dim*4)
        self.bn5 = nn.BatchNorm2d(self.dim*8)
        self.bn6 = nn.BatchNorm2d(self.dim*8)
        self.bn7 = nn.BatchNorm2d(self.dim*8)
        self.bn8 = nn.BatchNorm2d(self.dim*8)
        self.linear1 = self.linear = nn.Linear(4*4*4*2*self.dim, self.z)

    def forward(self, input):
        x = input.view(-1, *self.shape)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        x = F.elu(self.bn7(self.conv7(x)))
        x = F.elu(self.bn8(self.conv8(x)))
        x = x.view(-1, 4*4*4*2*self.dim)
        x = self.linear1(x)
        return x


class ToyEncoder(nn.Module):
    def __init__(self, args):
        super(ToyEncoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(1, self.dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2)
        self.linear1 = nn.Linear(4*4*4*self.dim, self.z)

    def forward(self, x):
        x = x.view(-1, *self.shape)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        return x
