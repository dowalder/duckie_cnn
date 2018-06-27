#!/usr/bin/env python3
import argparse
import os
import pathlib
import random
import yaml
from typing import Dict
import collections

import torch
import torch.nn
import torch.utils.data

import networks
import dataset


class Statistics:

    def __init__(self, path: pathlib.Path):
        self.train = collections.OrderedDict()
        self.test = collections.OrderedDict()
        self.path = path

    def add_train(self, value: float, iteration: int):
        self.train[iteration] = value

    def add_test(self, value: float, iteration: int):
        self.test[iteration] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.path.write_text(yaml.dump({"test": self.test, "train": self.train}))


class Params:

    def __init__(self, args):
        conf = yaml.load(pathlib.Path(args.conf).read_text())
        check_conf(conf)

        self.train_path = pathlib.Path(conf["train_path"])
        self.test_path = pathlib.Path(conf["test_path"])
        self.model_path = pathlib.Path(conf["model_path"])
        self.device = torch.device(conf["device"])
        self.network = conf["network"]
        self.num_epochs = conf["num_epochs"]
        self.pretrained = conf["pretrained"]

        self.test_interval = conf["intervals"]["test"]
        self.display_interval = conf["intervals"]["display"]
        self.save_interval = conf["intervals"]["save"]


def check_conf(conf: Dict):
    required_fields = {
        "train_path": str,
        "test_path": str,
        "model_path": str,
        "device": str,
        "intervals": dict,
        "network": str,
        "num_epochs": int,
        "pretrained": bool
    }
    for key, val in required_fields.items():
        assert key in conf, "Missing key: {}".format(key)
        assert isinstance(conf[key], val), "Expected {} to be {}, but got {}".format(key, val, conf[key].__class__)


def net_factory(net: str, params) -> torch.nn.Module:
    if net == "conv_rnn":
        return networks.BasicConvRNN(device=params.device)
    elif net == "resnet_rnn":
        return networks.ResnetRNN(pretrained=params.pretrained, device=params.device)
    elif net == "resnet_rnn_small":
        return networks.ResnetRNNsmall()
    else:
        raise RuntimeError("Unkown network: {}".format(net))


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


def train_rnn(params: Params):
    img_size = (224, 224) if params.network == "resnet_rnn" else (120, 160)
    train_set = dataset.RNNDataSet(params.train_path, 10, device=params.device, img_size=img_size)
    test_set = dataset.RNNDataSet(params.test_path, 10, device=params.device, img_size=img_size)

    net = net_factory(params.network, params)
    net.to(params.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(net.parameters())

    step = 0
    running_loss = 0

    with Statistics(params.model_path / "stat.yaml") as statistics:
        for epoch in range(params.num_epochs):
            for idx in range(len(train_set)):
                optimizer.zero_grad()
                net.zero_grad()

                net.init_hidden()

                idx_actually = random.choice(list(range(len(train_set))))

                imgs, actions, lbls = train_set[idx_actually]

                out = net(imgs, actions)

                out = out.squeeze()
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if step % params.display_interval == 0:
                    err = running_loss / params.display_interval
                    print("[{}][{}]: {}".format(epoch, idx, err))
                    statistics.add_train(err, step)
                    running_loss = 0

                if step % params.test_interval == 0:
                    with torch.no_grad():
                        test_loss = 0
                        net.init_hidden()
                        for imgs, actions, lbls in test_set:
                            out = net(imgs, actions)
                            out = out.squeeze()
                            loss = criterion(out, lbls)
                            test_loss += loss.item()

                        err = test_loss / len(test_set)
                        print("test: {}".format(err))
                        statistics.add_test(err, step)

                if step % params.save_interval == 0:
                    model_path = params.model_path / "step_{}.pth".format(step)
                    print("Saving model to {}".format(model_path))
                    torch.save(net.state_dict(), model_path.as_posix())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", help="configuration file (.yaml)", required=True)

    args = parser.parse_args()
    params = Params(args)

    train_rnn(params)


if __name__ == "__main__":
    main()

