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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR ,SequentialLR

train_paths = [
    "datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
]

# test_paths = [
#     "datasets/datasets_classification_processed/checxpert_data/test/",
# ]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def main(args):
    torch.hub.set_dir("/home/yandex/MLFH2023/eranlevin/hiera/")
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
    # dataset_test = FolderDataset(
    #     paths=test_paths,
    # )
    logging.info(f"Train dataset size: {len(dataset_train)}")
    # logging.info(f"Test dataset size: {len(dataset_test)}")
    dataloader_train = DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=4
    )
    # dataloader_test = DataLoader(
    #     dataset_test, batch_size=64, shuffle=True, num_workers=4
    # )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Assuming you have already defined your optimizer and dataloader
    total_epochs = 40
    num_batches = len(dataloader_train)


    # Combine both schedulers using SequentialLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=num_batches, epochs=total_epochs)
    ACCUMULATE_STEP = 4
    for epoch in range(40):
        accumulated_loss = 0.0  # Initialize accumulated loss for this epoch

        for batch_idx, batch in enumerate(tqdm(dataloader_train)):
            batch = batch.to(device)
            loss = model.forward(batch)[0]

            # Accumulate the loss
            accumulated_loss += loss

            if (batch_idx + 1) % ACCUMULATE_STEP == 0:
                # Backpropagate and update only after accumulating gradients for a certain number of minibatches
                accumulated_loss /= ACCUMULATE_STEP
                optimizer.zero_grad()
                accumulated_loss.backward()
                optimizer.step()
                scheduler.step()

                if args.log_wandb:
                    wandb.log({"loss": accumulated_loss, "learning_rate": scheduler.get_last_lr()[0]})

                accumulated_loss = 0.0  # Reset accumulated loss

        # Handle remaining accumulated gradients at the end of the epoch
        if (batch_idx + 1) % ACCUMULATE_STEP != 0:
            accumulated_loss /= ((batch_idx + 1) % ACCUMULATE_STEP)
            optimizer.zero_grad()
            accumulated_loss.backward()
            optimizer.step()
            scheduler.step()

            if args.log_wandb:
                wandb.log({"loss": accumulated_loss, "learning_rate": scheduler.get_last_lr()[0]})

    model.eval()
    loss_avg = torch.zeros(1)
    # for batch in tqdm(dataloader_test):
    #     loss = model.forward(batch)[0]
    #     loss_avg += loss

    # loss_avg /= len(dataloader_test)
    # if args.log_wandb:
    #     wandb.log({"evaluation_avg_loss": loss_avg})

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
    parser.add_argument("--log_wandb", type=bool, default=True)

    args = parser.parse_args()
    main(args)
