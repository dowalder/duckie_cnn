#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet

import resnet


def _num_flat_features(x):
    """
    Compute the number of features in a tensor.
    :param x:
    :return:
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)


class InitialNet(nn.Module):
    """
    The exact (as possible) copy of the caffe net in https://github.com/syangav/duckietown_imitation_learning
    """

    def __init__(self):
        super(InitialNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2)

        self.fc1 = nn.Linear(10 * 5 * 64, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, _num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NImagesNet(InitialNet):
    """
    This extends the InitialNet so that it accepts larger images. Images for this network can have n times more pixels
    in the height, but need to keep the original width.
    """

    def __init__(self, n):
        super(NImagesNet, self).__init__()
        self.fc1 = nn.Linear(n * 10 * 5 * 64, 1024)


class FutureLabelNet(NImagesNet):
    """
    Is able to give out the current and the next label as an output of the network.
    """

    def __init__(self, n):
        super(FutureLabelNet, self).__init__(n=n)
        self.fc2 = nn.Linear(1024, 2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicConvRNN(nn.Module):

    def __init__(self, device="cpu"):
        super(BasicConvRNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 11 * 16, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)

        self.num_lstms = 1
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(BasicConvRNN, self).to(device)
        self.device = device

    def cnn_pass(self, img):
        out = F.relu(self.conv1(img))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def rnn_pass(self, x, action):
        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        out = torch.cat((x, action), 1)
        out = out.unsqueeze(1)
        out, _ = self.rnn(out, self.hidden)
        out = self.fc_final(out)
        return out

    def forward(self, img, action):
        out = self.cnn_pass(img)
        out = self.rnn_pass(out, action)
        return out


class ResnetRNN(nn.Module):

    def __init__(self, pretrained=False, device="cpu"):
        super(ResnetRNN, self).__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=pretrained)

        self.num_lstms = 1
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(ResnetRNN, self).to(device)
        self.device = device

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def forward(self, img, action):
        x = self.resnet(img)

        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        x = torch.cat((x, action), 1)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x, self.hidden)
        x = self.fc_final(x)
        return x


class ResnetRNNsmall(nn.Module):

    def __init__(self, pretrained=False, device="cpu"):
        super(ResnetRNNsmall, self).__init__()
        self.num_lstms = 1

        self.resnet = resnet.resnet8()
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(input_size=1002, hidden_size=128, num_layers=self.num_lstms)
        self.fc_final = nn.Linear(in_features=128, out_features=2)

        self.device = device
        self.init_hidden()

    def to(self, device):
        super(ResnetRNNsmall, self).to(device)
        self.device = device

    def init_hidden(self):
        self.hidden = (torch.zeros(self.num_lstms, 1, 128, device=self.device),
                       torch.zeros(self.num_lstms, 1, 128, device=self.device))

    def forward(self, img, action):
        x = self.resnet(img)
        x = self.dropout(x)

        if len(action.shape) <= 1:
            action = action.unsqueeze(0)

        x = torch.cat((x, action), 1)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x, self.hidden)
        x = self.fc_final(x)
        return x
