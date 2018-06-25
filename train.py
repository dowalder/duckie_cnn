#!/usr/bin/env python3
import os
import pathlib
import random

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


def train_cnn(net,
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
                print("[%d][%d] training loss: %f" % (epoch, step, running_loss / disp_interval))
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


def train_rnn():
    train_sets = []
    # train_sets.append(dataset.RNNDataSet(pathlib.Path("/home/dominik/dataspace/images/randomwalk_forward/train_large"),
    #                                      10,
    #                                      device="cuda:0"))
    train_sets.append(dataset.RNNDataSet(pathlib.Path("/home/dwalder/data/images/randomwalk_forward/train_large"),
                                         10,
                                         device="cuda:0"))
    test_set = dataset.RNNDataSet(pathlib.Path("/home/dwalder/data/images/randomwalk_forward/test"),
                                  10,
                                  device="cuda:0")

    test_interval = 500
    save_interval = 5000
    display_interval = 250

    net = networks.BasicConvRNN()
    net.to("cuda:0")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(net.parameters())

    num_epochs = 100
    step = 0
    running_loss = 0

    for train_set in train_sets:
        for epoch in range(num_epochs):
            for idx in range(len(train_set)):
                optimizer.zero_grad()
                net.zero_grad()

                net.init_hidden()

                idx_actually = random.choice(list(range(len(train_set))))

                imgs, actions, lbls = train_set[idx_actually]

                out = net.cnn_pass(imgs)
                out = net.rnn_pass(out, actions)

                out = out.squeeze()
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if step % display_interval == 0:
                    print("[{}][{}]: {}".format(epoch, idx, running_loss / display_interval))
                    running_loss = 0

                if step % test_interval == 0:
                    with torch.no_grad():
                        test_loss = 0
                        net.init_hidden()
                        for imgs, actions, lbls in test_set:
                            out = net(imgs, actions)
                            out = out.squeeze()
                            loss = criterion(out, lbls)
                            test_loss += loss.item()

                        print("test: {}".format(test_loss / len(test_set)))

                if step % save_interval == 0:
                    model_path = pathlib.Path(
                        "/home/dwalder/data/models/rnn_randomwalk_forward") / "step_{}.pth".format(step)
                    print("Saving model to {}".format(model_path))
                    torch.save(net.state_dict(), model_path.as_posix())


if __name__ == "__main__":
    train_rnn()

