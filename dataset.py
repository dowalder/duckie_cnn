#!/usr/env/python
import os
import PIL.Image

import torch
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import cv2


class DataSet(torch.utils.data.Dataset):
    """
    A simple data set to use when you have a single folder containing the images and for every image a .txt file in the
    same directory with the same name containing a single line with space separated values as labels.
    """

    def __init__(self, data_dir):
        self.images = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir)
                       if img_file.endswith(".jpg")]
        self.labels = []
        for image in self.images:
            stem, _ = os.path.splitext(os.path.basename(image))
            lbl_file = os.path.join(os.path.dirname(image), "{}.txt".format(stem))
            if not os.path.isfile(lbl_file):
                raise IOError("Could not find the label file {}".format(lbl_file))
            with open(lbl_file, "r") as fid:
                self.labels.append(list(map(float, fid.read().strip().split(" "))))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = PIL.Image.open(self.images[item])
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return torch.Tensor(self.labels[item]), self.transform(img)


class NImagesDataSet(DataSet):
    """
    And extended data set, that returns n images concatenated as one instead of a single one.
    """

    def __init__(self, data_dir, n=1):
        super(NImagesDataSet, self).__init__(data_dir)

        self.n = n
        self.images.sort()

        if len(self) < self.n:
            raise RuntimeError("Found {} images in {}, but require at least {}.".format(len(self), data_dir, self.n))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((80 * self.n, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images) - self.n

    def __getitem__(self, item):
        imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
        img = np.concatenate(tuple(imgs))
        return torch.Tensor(self.labels[item + self.n]), self.transform(img)


class FutureLabelDataSet(NImagesDataSet):

    def __len__(self):
        # for every entry, we need a future label, therefor we cannot return a value for the last image/label pair
        return len(self.images) - self.n - 1

    def __getitem__(self, item):
        imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
        img = np.concatenate(tuple(imgs))
        return torch.Tensor(self.labels[item + self.n] + self.labels[item + self.n + 1]), self.transform(img)


if __name__ == "__main__":
    pass
