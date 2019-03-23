from torch import nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm as SN


class CELEBAdiscriminator(nn.Module):
    def __init__(self, args):
        super(CELEBAdiscriminator, self).__init__()
        self._name = 'cifarD'
        self.shape = (64, 64, 3)
        self.dim = args.dim

        self.conv1 = SN(nn.Conv2d(3, self.dim, 3, 1, padding=1))
        self.conv2 = SN(nn.Conv2d(self.dim, self.dim, 3, 2, padding=1))
        self.conv3 = SN(nn.Conv2d(self.dim, 2 * self.dim, 3, 1, padding=1))
        self.conv4 = SN(nn.Conv2d(2 * self.dim, 2 * self.dim, 3, 2, padding=1))
        self.conv5 = SN(nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 1, padding=1))
        self.conv6 = SN(nn.Conv2d(4 * self.dim, 4 * self.dim, 3, 2, padding=1))
        self.linear = SN(nn.Linear(4*4*4*self.dim, 1))

    def forward(self, input):
        input = input.view(-1, 3, 64, 64)
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        output = x.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output


class CIFARdiscriminator(nn.Module):
    def __init__(self, args):
        super(CIFARdiscriminator, self).__init__()
        self._name = 'cifarD'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 3, 32, 32)
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output


class MNISTdiscriminator(nn.Module):
    def __init__(self, args):
        super(MNISTdiscriminator, self).__init__()
        self._name = 'mnistD'
        self.shape = (1, 28, 28)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1)
