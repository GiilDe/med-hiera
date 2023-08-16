from random import shuffle
from typing import Optional
import numpy as np
import torch
from torchvision.datasets import VisionDataset
import PIL.Image
import os
from glob import glob
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class FolderDataset(VisionDataset):
    labels = None
    original_size = (600, 450)  # size of the example image in the hiera repo
    input_size = 224

    prefix_transform = [
        transforms.Resize(
            int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]
    normalize_chesxray = [
        transforms.Normalize((0.5317, 0.5317, 0.5317), (0.2001, 0.2001, 0.2001)),
    ]
    normalize_all_data = [
        transforms.Normalize((0.5505, 0.5220, 0.5247), (0.1894, 0.1934, 0.1953)),
    ]
    default_transform = transforms.Compose(
        prefix_transform + normalize_all_data
    )


    @staticmethod
    def process_labels(labels_path):
        if FolderDataset.labels is None:
            FolderDataset.labels = pd.read_csv(labels_path)[
                ["Image Index", "Finding Labels"]
            ]
            FolderDataset.labels["Finding Labels"] = pd.DataFrame.apply(
                FolderDataset.labels[["Finding Labels"]],
                lambda row: row["Finding Labels"].split("|"),
                axis=1,
            )
            FolderDataset.labels = FolderDataset.labels.explode("Finding Labels")
            FolderDataset.labels = pd.get_dummies(
                FolderDataset.labels, columns=["Finding Labels"]
            )
            FolderDataset.labels = FolderDataset.labels.groupby(
                by=["Image Index"]
            ).any()
            FolderDataset.labels = FolderDataset.labels.to_dict("index")

    def __init__(self, paths, transform=default_transform, labels_path=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """
        if not isinstance(paths, list):
            paths = [paths]
        self.img_paths = []
        for path in paths:
            self.img_paths += list(glob(path + "*.jpg", recursive=True)) + list(
                glob(path + "*.png", recursive=True)
            )
        self.labels = None
        if labels_path is not None:
            FolderDataset.process_labels(labels_path)

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
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        X = self.get_image_from_folder(img_path)
        X = X.convert("RGB")
        # X = X.resize(size=FolderDataset.original_size)
        X = self.transform(X)
        if FolderDataset.labels is not None:
            labels = list(FolderDataset.labels[img_name].values())
            label_torch = torch.zeros(size=(len(labels),))
            label_torch[labels] = 1
            return (X, label_torch)
        return X
