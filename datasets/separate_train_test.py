from glob import glob
import os
import shutil
from tqdm import tqdm

test_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/"
validation_1_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/validation_1/"
validation_2_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/validation_2/"
train_folder = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/"


if __name__ == "__main__":
    os.makedirs(os.path.dirname(test_folder), exist_ok=True)
    os.makedirs(os.path.dirname(train_folder), exist_ok=True)

    with open(
        "datasets/datasets_classification/checxpert_data/test_list.txt", "r"
    ) as f:
        test_paths = f.read()
        test_paths = test_paths.split("\n")
        test_paths = list(test_paths)
        ind1 = int(len(test_paths) / 3)
        ind2 = int((2 * len(test_paths)) / 3)
        val_1_paths = set(test_paths[:ind1])
        val_2_paths = set(test_paths[ind1:ind2])
        test_paths_ = set(test_paths[ind2:])

    images_paths = list(
        glob("datasets/datasets_classification/checxpert_data/**/*.png", recursive=True)
    ) + list(
        glob("datasets/datasets_classification/checxpert_data/**/*.jpg", recursive=True)
    )
    print("test size", len(test_paths_))
    print("val1 size", len(val_1_paths))
    print("val2 size", len(val_2_paths))
    print("total size", len(images_paths))
    for path in tqdm(images_paths):
        img_name = os.path.basename(path)
        if img_name in test_paths_:
            shutil.copyfile(path, os.path.join(test_folder, img_name))
        elif img_name in val_1_paths:
            shutil.copyfile(path, os.path.join(validation_1_folder, img_name))
        elif img_name in val_2_paths:
            shutil.copyfile(path, os.path.join(validation_2_folder, img_name))
        else:
            shutil.copyfile(path, os.path.join(train_folder, img_name))
