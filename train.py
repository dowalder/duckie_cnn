#!/usr/bin/env python
import os

import torch
import torch.nn
import torch.utils.data

import networks
import dataset


def validation(net, test_loader, criterion, device="cpu"):
    """
    Perform a validation step.

    :param net: torch.nn.Module -> the neural network
    :param test_loader: torch.utils.data.DataLoader -> the validation data
    :param criterion:
    :param device:
    """
    avg_mse = 0
    for data in test_loader:
        labels, images = data
        labels = labels.to(device)
        images = images.to(device)

        outputs = net(images)

        loss = criterion(outputs, labels)
        avg_mse += loss.item()
    avg_mse /= len(test_loader)

    print("\ttest loss: %f" % avg_mse)


def train_net(net,
              train_loader,
              test_loader,
              criterion,
              optimizer,
              save_dir,
              device="cpu",
              num_epoch=100,
              disp_interval=10,
              val_interval=50,
              save_interval=20):
    """
    Training a network.

    :param net: The pytorch network. It should be initialized (as not initialization is performed here).
    :param train_loader: torch.data.utils.DataLoader -> to train the classifier
    :param test_loader: torch.data.utils.DataLoader -> to test the classifier
    :param criterion: see pytorch tutorials for further information
    :param optimizer: see pytorch tutorials for further information
    :param save_dir: str -> where the snapshots should be stored
    :param device: str -> "cpu" for computation on CPU or "cuda:n",
                            where n stands for the number of the graphics card that should be used.
    :param num_epoch: int -> number of epochs to train
    :param disp_interval: int -> interval between displaying training loss
    :param val_interval: int -> interval between performing validation
    :param save_interval: int -> interval between saving snapshots
    """
    save_dir = os.path.expanduser(save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    print("Moving the network to the device {}...".format(device))
    net.to(device)

    step = 0
    running_loss = 0
    print("Starting training")
    for epoch in range(num_epoch):
        for lbls, imgs in train_loader:

            optimizer.zero_grad()

            lbls = lbls.to(device)
            imgs = imgs.to(device)

            outputs = net(imgs)

            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % disp_interval == 0 and step != 0:
                print("[%d][%d] training loss: %f" % (epoch, step, running_loss / val_interval))
                running_loss = 0

            if step % val_interval == 0 and step != 0:
                print("[%d][%d] Performing validation..." % (epoch, step))
                validation(net, test_loader, criterion=criterion, device=device)

            if step % save_interval == 0 and epoch != 0:
                path = os.path.join(save_dir, "checkpoint_{}.pth".format(step))
                print("[%d][%d] Saving a snapshot to %s" % (epoch, step, path))
                torch.save(net.state_dict(), path)

            step += 1


def exact_caffe_copy_factory(train_path, test_path):
    """
    Prepare the training in such a way that the caffe net proposed in
    https://github.com/syangav/duckietown_imitation_learning is copied in pytorch.

    :param train_path: str -> path to training data
    :param test_path: str -> path to testing data
    :return:
    """
    train_set = dataset.DataSet(train_path)
    test_set = dataset.DataSet(test_path)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    net = networks.InitialNet()
    net.apply(networks.weights_init)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.85, weight_decay=0.0005)

    return net, train_loader, test_loader, criterion, optimizer
