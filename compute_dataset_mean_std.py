import torch
from utils import FolderDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms


train_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
]

if __name__ == "__main__":
    dataset_train = FolderDataset(
        paths=train_paths, transform=transforms.Compose(FolderDataset.prefix_transform)
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4
    )
    mean = 0.
    std = 0.
    nb_samples = 0.
    j = 0
    for data in tqdm(dataloader_train):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        j += batch_samples
        if j >= 10000:
            break

    mean /= nb_samples
    std /= nb_samples
    print(mean, std)