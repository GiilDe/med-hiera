from torch import optim
from torch.utils.data import DataLoader
import torch
from hiera.hiera import Hiera, hiera_tiny_224
import torchvision
from utils import FolderDataset
import wandb
import logging
from tqdm import tqdm
import argparse
from torch.nn import BCELoss, Sigmoid

path = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification/**/"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
def main(args):
    model: Hiera = hiera_tiny_224(pretrained=False, checkpoint="hiera_in1k", num_classes=15)
    # strict=False because the .pth includes the decoder params which we don't need for classification.
    # Plus, the .pth file lacks some params that are needed in the model
    model.load_state_dict(torch.load("med-mae_hiera_tiny_224.pth"), strict=False) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.log_wandb:
        wandb.login()
        wandb.init(
            name="med-hiera_1 classification",
            # Set the project where this run will be logged
            project="med-hiera",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
            },
        )

    dataset = FolderDataset(
        path=path,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        labels_path="/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification/checxpert_data/Data_Entry_2017.csv",
    )
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4
    )
    sigmoid = Sigmoid()
    loss_func = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model = model.to(device)
    model.train()

    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        predictions = model.forward(x)
        predictions = sigmoid(predictions)
        optimizer.zero_grad()
        loss = loss_func(predictions, y)
        loss.backward()
        optimizer.step()

        if args.log_wandb:
            wandb.log({"loss": loss})


    model.eval()
    if args.save_model:
        torch.save(model.state_dict(), "med-mae_hiera_tiny_224_classification.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a customizable learning rate.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model training')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--log_wandb', type=bool, default=False)

    args = parser.parse_args()
    main(args)
