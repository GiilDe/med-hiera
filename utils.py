from random import shuffle
from torchvision.datasets import VisionDataset
import PIL.Image
import os
from glob import glob


class RepeatChannel:
    def __init__(self):
        self.n_channels = 3

    def __call__(self, X):
        if X.shape[0] == 1:
            X = X.repeat(self.n_channels, 1, 1)
        return X


class FolderDataset(VisionDataset):
    def __init__(self, path="data", transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """
        self.img_paths = [
            path
            for path in glob(path + "*.jpg", recursive=True)
            + glob(path + "*.png", recursive=True)
        ]
        shuffle(self.img_paths)
        self.transform = transform

    def get_image_from_folder(self, path):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = PIL.Image.open(path)
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """
        X = self.get_image_from_folder(self.img_paths[index])
        X = X.convert("RGB")
        X = self.transform(X)
        return X
