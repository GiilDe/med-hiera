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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

train_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
]

test_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/",
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
    device = torch.device("cuda")
    model = model.to(device)
    if args.log_wandb:
        wandb.login()
        wandb.init(
            name=args.wandb_run_name if args.wandb_run_name else "med-hiera_1",
            # Set the project where this run will be logged
            project="med-hiera",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
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
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Assuming you have already defined your optimizer and dataloader
    ACCUMULATION_STEPS = 1
    num_batches = int(len(dataloader_train) / ACCUMULATION_STEPS)

    # Combine both schedulers using SequentialLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=num_batches,
        epochs=args.epochs,
    )
    try:
        for epoch in range(args.epochs):
            model.train()
            for batch_idx, batch in enumerate(tqdm(dataloader_train)):
                batch = batch.to(device)
                loss = model.forward(batch)[0]
                loss /= ACCUMULATION_STEPS
                loss.backward()
                if args.log_wandb:
                    wandb.log(
                        {"loss": loss, "learning_rate": scheduler.get_last_lr()[0]}
                    )

                if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (
                    batch_idx + 1 == len(dataloader_train)
                ):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                loss_avg = torch.zeros(1).to(device)
                for batch in tqdm(dataloader_test):
                    batch = batch.to(device)
                    loss = model.forward(batch)[0]
                    loss_avg += loss

                loss_avg /= len(dataloader_test)
                if args.log_wandb:
                    wandb.log({"evaluation_avg_loss": loss_avg})

                logging.info(f"Average loss: {loss_avg}")
    finally:
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
    parser.add_argument("--log_wandb", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wandb_run_name", type=str, default="")

    args = parser.parse_args()
    main(args)
