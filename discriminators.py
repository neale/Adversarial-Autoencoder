from torch import nn

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
