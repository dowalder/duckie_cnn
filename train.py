#!/usr/env/python3

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim

import dataset


DEVICE = "cuda:1"


def _num_flat_features(x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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

    def __init__(self, n: int = 1):
        super(NImagesNet, self).__init__()
        self.fc1 = nn.Linear(n * 10 * 5 * 64, 1024)


def validation(net: InitialNet, test_loader: torch.utils.data.DataLoader) -> None:
    """
    Perform a validation step on the test set loaded by test_loader.

    :param net:
    :param test_loader:
    :return:
    """
    avg_mse = 0
    for data in test_loader:
        labels, images = data
        labels = labels.to(DEVICE)
        images = images.to(DEVICE)

        outputs = net(images)

        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        avg_mse += loss.item()
    avg_mse /= len(test_loader)

    print("test: {}".format(avg_mse))


def main():
    print("Loading data...")
    train_set = dataset.NImagesDataSet(pathlib.Path("train_images"), n=3)
    test_set = dataset.NImagesDataSet(pathlib.Path("test_images"), n=3)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=10)

    print("Loading net...")
    net = NImagesNet(n=3)
    net.apply(weights_init)
    print("To device...")
    net.to(DEVICE)

    print("Loading optimizers...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.85, weight_decay=0.0005)

    running_loss = 0

    print("Started training:")
    for epoch in range(100):
        # Perform learning rate decay
        if epoch == 15:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.85, weight_decay=0.0005)

        for i, (lbls, imgs) in enumerate(train_loader):
            optimizer.zero_grad()

            lbls = lbls.to(DEVICE)
            imgs = imgs.to(DEVICE)

            outputs = net(imgs)

            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 4:
                print("train [{}/{}]: {}".format(i, epoch, running_loss / 10))
                running_loss = 0
                validation(net, test_loader)


if __name__ == "__main__":
    main()
