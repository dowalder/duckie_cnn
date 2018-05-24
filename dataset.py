#!/usr/env/python3

import pathlib

import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL.Image


class DataSet(torch.utils.data.Dataset):
    """
    A simple data set to use when you have a single folder containing the images and for every image a .txt file in the
    same directory with the same name containing a single line with space separated values as labels.
    """

    def __init__(self, data_dir: pathlib.Path):
        self.images = list(data_dir.glob("*.jpg"))
        self.labels = []
        for image in self.images:
            lbl_file = image.parent / "{}.txt".format(image.stem)
            if not lbl_file.is_file():
                raise FileNotFoundError("Could not find the label file {}".format(lbl_file))
            self.labels.append(list(map(float, lbl_file.read_text().strip().split(" "))))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = PIL.Image.open(self.images[item].as_posix())
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return torch.Tensor(self.labels[item]), self.transform(img)


if __name__ == "__main__":
    pass
