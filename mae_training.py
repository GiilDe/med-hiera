from torch import optim, utils
import torch
from hiera.hiera_mae import MaskedAutoencoderHiera
import torchvision
from utils import FolderDataset
import wandb
import logging
from tqdm import tqdm
import argparse

path = "/home/yandex/MLFH2023/giladd/hiera/datasets/**/"

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

    wandb.login()
    run = wandb.init(
        name="med-hiera_1",
        # Set the project where this run will be logged
        project="med-hiera",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
        },
    )

    #    wandb.log_artifact(model, type="model")

    dataset = FolderDataset(
        path=path,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5, last_epoch=-1
    )
    model.train()

    for epoch in range(10):
        for batch in tqdm(dataloader):
            loss = model.forward(batch)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss, "learning_rate": scheduler.get_lr()[0]})

    model.eval()