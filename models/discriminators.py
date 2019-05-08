import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallDiscriminator(nn.Module):
    def __init__(self, args):
        super(SmallDiscriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(self.nc, self.dim, 2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, x):
        x = x.view(-1, *self.shape)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        return x
 

class DeepDiscriminator(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(DeepDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, self.dim, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(self.dim, self.dim, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(self.dim, 2*self.dim, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(2*self.dim, 2*self.dim, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(2*self.dim, 4*self.dim, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(4*self.dim, 4*self.dim, 3, 2, padding=1)
        self.linear = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, x):
        x = x.view(-1, *self.shape)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear(x)
        return x

class ToyDiscriminator(nn.Module):
    def __init__(self, args):
        super(ToyDiscriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(1, self.dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2)
        self.linear1 = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x), inplace=True)
        x = F.elu(self.conv2(x), inplace=True)
        x = F.elu(self.conv3(x), inplace=True)
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        return x.view(-1)

