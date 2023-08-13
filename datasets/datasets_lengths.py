from glob import glob
import os
import shutil

datasets_paths = [
    "datasets/datasets_classification_processed/checxpert_data/test/",
    "datasets/datasets_classification_processed/checxpert_data/train/",
    "datasets/datasets_mae/**/",
    "datasets/datasets_classification/checxpert_data/**/",
]


if __name__ == "__main__":
    for path in datasets_paths:
        images_paths = list(glob(path + "*.jpg", recursive=True)) + list(
            glob(path + "*.png", recursive=True)
        )
        print(path, len(images_paths))
