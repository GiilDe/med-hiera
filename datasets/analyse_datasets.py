from glob import glob
import os
import shutil
import PIL.Image

datasets_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/covid_data/COVID-19_Radiography_Dataset/**/"
]


if __name__ == "__main__":
    for path in datasets_paths:
        images_paths = list(glob(path + "*.jpg", recursive=True)) + list(
            glob(path + "*.png", recursive=True)
        )
        print(path)
        print("dataset size", len(images_paths))
        image = PIL.Image.open(images_paths[0])
        print("image size", image.size)
        print("image format", image.format)
        print("image mode", image.mode)
    
    example_image_path = "examples/img/dog.jpeg" 
    print("example image path", example_image_path)
    print("image size", image.size)
    print("image format", image.format)
    print("image mode", image.mode)
