import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class myResNetImg(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResNetImg, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        att = F.adaptive_avg_pool2d(x, [att_size, att_size])

        if not self.if_fine_tune:
            att = Variable(att.data)

        return att


class myResNetRoI(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResNetRoI, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)

        if not self.if_fine_tune:
            fc = Variable(fc.data)

        return fc
