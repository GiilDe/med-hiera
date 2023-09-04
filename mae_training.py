from torch import optim
from torch.utils.data import DataLoader
import torch
from hiera.hiera_mae import MaskedAutoencoderHiera
import torchvision
from utils import FolderDataset
import logging
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.transforms import ToTensor
from rand_augment import RandAugment
from distutils.util import strtobool

train_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/",
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/**/",
    # "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/covid_cxr2/**/",
    # "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_mae/COVID-19_Radiography_Dataset/**/",
]

test_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/validation_1/",
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def main(args):
    torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
    pretrained_hub = (not args.init_random_weights) and (
        args.pretrained_model_path == ""
    )
    model: MaskedAutoencoderHiera = torch.hub.load(
        "facebookresearch/hiera",
        model="mae_hiera_small_224",
        pretrained=pretrained_hub,
        checkpoint="mae_in1k",
    )
    logging.info(f"Model loaded from torch.hub: {pretrained_hub}")
    if args.pretrained_model_path != "":
        logging.info(f"Loading pretrained model from {args.pretrained_model_path}")
        model.load_state_dict(torch.load(args.pretrained_model_path))
    device = torch.device("cuda")
    model = model.to(device)
    if args.log_wandb:
        import wandb

        wandb.login()
        wandb.init(
            entity="med-hiera",
            name=args.wandb_run_name if args.wandb_run_name else "med-hiera_1",
            # Set the project where this run will be logged
            project="med-hiera",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "mask_ratio": args.mask_ratio,
                "use_augmentations": args.use_augmentations,
                "use_cocktail": args.use_cocktail,
                "init_random_weights": args.init_random_weights,
                "use_main_data": args.use_main_data,
            },
        )

    train_transform = FolderDataset.prefix_transform[:-1].copy()
    train_transform.append(RandAugment())
    train_transform.append(ToTensor())
    train_transform += FolderDataset.normalize_all_data
    train_transform = torchvision.transforms.Compose(train_transform)

    train_paths_ = []
    if args.use_main_data:
        train_paths_.append(train_paths[0])
    if args.use_cocktail:
        train_paths_ += train_paths[1:]
    dataset_train = FolderDataset(
        paths=train_paths_,
        transform=train_transform
        if args.use_augmentations
        else FolderDataset.default_transform,
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
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(dataloader_train)):
            batch = batch.to(device)
            loss = model.forward(batch, mask_ratio=args.mask_ratio)[0]
            loss /= ACCUMULATION_STEPS
            loss.backward()
            if args.log_wandb:
                wandb.log(
                    {
                        "loss": loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epochs": args.epochs,
                    }
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
    if args.save_model:
        torch.save(model.state_dict(), args.save_model_name)


def init_args():
    parser = argparse.ArgumentParser(
        description="Train a model with a customizable learning rate."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for model training",
    )
    parser.add_argument(
        "--save_model", type=lambda x: bool(strtobool(x)), default=False
    )
    parser.add_argument("--log_wandb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--save_model_name", type=str, default="med-mae_hiera_tiny_224.pth"
    )
    parser.add_argument("--weight_decay", type=float, default=0)  # 1e-8
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument(
        "--use_augmentations", type=lambda x: bool(strtobool(x)), default=False
    )
    parser.add_argument(
        "--use_cocktail", type=lambda x: bool(strtobool(x)), default=True
    )
    parser.add_argument(
        "--init_random_weights", type=lambda x: bool(strtobool(x)), default=False
    )
    parser.add_argument(
        "--use_main_data", type=lambda x: bool(strtobool(x)), default=True
    )
    parser.add_argument("--pretrained_model_path", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    assert args.use_main_data or args.use_cocktail
    main(args)
