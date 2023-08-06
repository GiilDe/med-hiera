from torch import optim, utils
import torch
from hiera.hiera_mae import MaskedAutoencoderHiera
import torchvision
from utils import FolderDataset
import wandb
import logging
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

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

    device = torch.device("cuda")
    model = model.to(device)

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
        dataset, batch_size=64, shuffle=True, num_workers=4
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Assuming you have already defined your optimizer and dataloader
    total_epochs = 40
    num_batches = len(dataloader)
    total_iters = total_epochs * num_batches

    # Create a LinearLR for linear warm-up
    warmup_epochs = 5
    warmup_iters = warmup_epochs * num_batches
    warmup_factor = 0.01
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.0, end_factor=warmup_factor, total_iters=warmup_iters)

    # Create a CosineAnnealingLR for cosine decay
    cosine_decay_epochs = total_epochs - warmup_epochs
    cosine_decay_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_decay_epochs * num_batches)

    # Combine both schedulers using SequentialLR
    scheduler = SequentialLR(optimizer, [warmup_lr_scheduler, cosine_decay_lr_scheduler])

    for epoch in range(40):
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            loss = model.forward(batch)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss, "learning_rate": scheduler.get_last_lr()[0]})

    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a customizable learning rate.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model training')

    args = parser.parse_args()
    main(args)
