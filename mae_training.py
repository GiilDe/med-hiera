from torch import optim
from torch.utils.data import DataLoader
import torch
from hiera.hiera_mae import MaskedAutoencoderHiera
import torchvision
from utils import FolderDataset
import wandb
import logging
from tqdm import tqdm
import argparse

train_paths = [
    "datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
]

test_paths = [
    "datasets/datasets_classification_processed/checxpert_data/test/",
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def main(args):
    torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
    model: MaskedAutoencoderHiera = torch.hub.load(
        "facebookresearch/hiera",
        model="mae_hiera_tiny_224",
        pretrained=True,
        checkpoint="mae_in1k",
    )
    if args.log_wandb:
        wandb.login()
        wandb.init(
            name="med-hiera_1",
            # Set the project where this run will be logged
            project="med-hiera",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
            },
        )

    dataset_train = FolderDataset(
        paths=train_paths,
    )
    dataset_test = FolderDataset(
        paths=test_paths,
    )
    logging.info(f"Train dataset size: {len(dataset_train)}")
    logging.info(f"Test dataset size: {len(dataset_test)}")
    dataloader_train = DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=4
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=32, shuffle=True, num_workers=4
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    for batch in tqdm(dataloader_train):
        loss = model.forward(batch)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log_wandb:
            wandb.log({"loss": loss})

    model.eval()
    loss_avg = torch.zeros(1)
    for batch in tqdm(dataloader_test):
        loss = model.forward(batch)[0]
        loss_avg += loss

    loss_avg /= len(dataloader_test)
    if args.log_wandb:
        wandb.log({"evaluation_avg_loss": loss_avg})

    logging.info(f"Average loss: {loss_avg}")
    if args.save_model:
        torch.save(model.state_dict(), "med-mae_hiera_tiny_224.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a customizable learning rate."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for model training",
    )
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--log_wandb", type=bool, default=False)

    args = parser.parse_args()
    main(args)
