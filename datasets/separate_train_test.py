from glob import glob
import os
import shutil
from tqdm import tqdm

test_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/"
train_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/"


if __name__ == "__main__":
    os.makedirs(os.path.dirname(test_folder), exist_ok=True)
    os.makedirs(os.path.dirname(train_folder), exist_ok=True)

    with open(
        "datasets/datasets_classification/checxpert_data/test_list.txt", "r"
    ) as f:
        test_paths = f.read()
        test_paths = test_paths.split("\n")
        test_paths = set(test_paths)

    images_paths = list(
        glob("datasets/datasets_classification/checxpert_data/**/*.png", recursive=True)
    ) + list(
        glob("datasets/datasets_classification/checxpert_data/**/*.jpg", recursive=True)
    )
    print("test size", len(test_paths))
    print("total size", len(images_paths))
    for path in tqdm(images_paths):
        img_name = os.path.basename(path)
        if img_name in test_paths:
            shutil.copyfile(path, os.path.join(test_folder, img_name))
        else:
            shutil.copyfile(path, os.path.join(train_folder, img_name))
