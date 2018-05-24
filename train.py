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


class BasicCopyNet(nn.Module):
    """
    The exact copy of the caffe net in https://github.com/syangav/duckietown_imitation_learning
    """

    def __init__(self):
        super(BasicCopyNet, self).__init__()

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


class ThreeImagesNet(BasicCopyNet):

    def __init__(self):
        super(ThreeImagesNet, self).__init__()
        self.fc1 = nn.Linear(30 * 5 * 64, 1024)


def validation(net: BasicCopyNet, test_loader: torch.utils.data.DataLoader) -> None:
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
    # train_set = dataset.ThreeImagesDataSet(pathlib.Path("/home/dominik/workspace/duckietown_imitation_learning/train_images"))
    # test_set = dataset.ThreeImagesDataSet(pathlib.Path("/home/dominik/workspace/duckietown_imitation_learning/test_images"))

    print("Loading data...")
    train_set = dataset.ThreeImagesDataSet(pathlib.Path("train_images"))
    test_set = dataset.ThreeImagesDataSet(pathlib.Path("test_images"))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=10)

    print("Loading net...")
    net = ThreeImagesNet()
    net.apply(weights_init)
    print("To device...")
    net.to(DEVICE)

    print("Loading optimizers...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.85, weight_decay=0.0005)

    running_loss = 0

    print("Started training:")
    for epoch in range(100):
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
