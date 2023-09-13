from torch import optim
from torch.utils.data import DataLoader
import torch
from hiera.hiera import Hiera, hiera_small_224, hiera_tiny_224
import torchvision
from utils import FolderDataset
import logging
from tqdm import tqdm
import argparse
from torch.nn import BCELoss, Sigmoid
from sklearn.metrics import roc_curve, auc
import numpy as np
from distutils.util import strtobool
from rand_augment import RandAugment
from torchvision.transforms import ToTensor, RandomRotation

train_paths = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/"
test_paths = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/validation_2/"
labels_path = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification/checxpert_data/Data_Entry_2017.csv"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

random_weights = "random-weights"


def main(args):
    if args.log_wandb:
        import wandb

        wandb.login()
        wandb.init(
            entity="med-hiera",
            name=args.wandb_run_name
            if args.wandb_run_name
            else "med-hiera_1 classification",
            # Set the project where this run will be logged
            project="med-hiera",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "pretrained_path": args.pretrained_path,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "use_augmentations": args.use_augmentations,
                "head_dropout": args.head_dropout,
                "size": args.size,
                "rotation_angle": args.rotation_angle,
            },
        )

    assert args.rotation_angle == 0 or args.use_augmentations is False
    if args.use_augmentations:
        train_transform = FolderDataset.prefix_transform[:-1].copy()
        train_transform.append(RandAugment())
        train_transform.append(ToTensor())
        train_transform += FolderDataset.normalize_all_data
        train_transform = torchvision.transforms.Compose(train_transform)
    else:
        train_transform = FolderDataset.prefix_transform[:-1].copy()
        train_transform.append(RandomRotation(args.rotation_angle))
        train_transform.append(ToTensor())
        train_transform += FolderDataset.normalize_all_data
        train_transform = torchvision.transforms.Compose(train_transform)

    if args.size == "tiny":
        model: Hiera = hiera_tiny_224(
            pretrained=False,
            checkpoint=None,
            num_classes=15,
            head_dropout=args.head_dropout,
        )
    else:
        model: Hiera = hiera_small_224(
            pretrained=False,
            checkpoint=None,
            num_classes=15,
            head_dropout=args.head_dropout,
        )
    if ".pth" in args.pretrained_path:
        model_state_dict = torch.load(args.pretrained_path)
        logging.info(f"Loaded model from path {args.pretrained_path}")
    elif args.pretrained_path != random_weights:
        torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
        url = "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth"
        model_state_dict = torch.hub.load_state_dict_from_url(url)["model_state"]
        logging.info(f"Loaded model from url {url}")
        if "head.projection.weight" in model_state_dict:
            del model_state_dict["head.projection.weight"]
        if "head.projection.bias" in model_state_dict:
            del model_state_dict["head.projection.bias"]

    if args.pretrained_path != random_weights:
        incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
        logging.info(
            f"Loaded model with missing keys: {incompatible_keys.missing_keys}"
        )
        logging.info(
            f"Loaded model with unexpected keys: {incompatible_keys.unexpected_keys}"
        )
        logging.info(
            f"number of incompatible keys: {len(incompatible_keys.missing_keys) + len(incompatible_keys.unexpected_keys)}"
        )
        logging.info(f"overall number of keys: {len(model_state_dict)}")
    else:
        logging.info("Loaded random weights")

    device = torch.device("cuda")

    dataset_train = FolderDataset(
        paths=train_paths,
        labels_path=labels_path,
        transform=train_transform
        if args.use_augmentations or args.rotation_angle != 0
        else FolderDataset.default_transform,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    dataset_test = FolderDataset(
        paths=test_paths,
        labels_path=labels_path,
    )
    logging.info(f"Train dataset size: {len(dataset_train)}")
    logging.info(f"Test dataset size: {len(dataset_test)}")
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    sigmoid = Sigmoid()
    loss_func = BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model = model.to(device)

    ACCUMULATION_STEPS = 1
    num_batches = int(len(dataloader_train) / ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=num_batches,
        epochs=args.epochs,
    )
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(dataloader_train):
            x = x.to(device)
            y = y.to(device)
            predictions = model.forward(x)
            predictions = sigmoid(predictions)
            optimizer.zero_grad()
            loss = loss_func(predictions, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.log_wandb:
                wandb.log({"loss": loss, "learning_rate": scheduler.get_last_lr()[0]})

        model.eval()
        with torch.no_grad():
            loss_avg = torch.zeros(1).to(device)
            all_predictions = torch.empty(size=(len(dataset_test), 15)).to(device)
            all_labels = torch.empty(size=(len(dataset_test), 15)).to(device)
            for i, batch in enumerate(tqdm(dataloader_test)):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                predictions = model.forward(x)
                predictions = sigmoid(predictions)
                i_ = i * args.batch_size
                j = i_ + x.shape[0]
                all_predictions[i_:j, :] = predictions
                all_labels[i_:j, :] = y
                loss = loss_func(predictions, y)
                loss_avg += loss

            loss_avg /= len(dataloader_test)

            all_predictions = all_predictions.cpu()
            all_labels = all_labels.cpu()
            auc_scores = []
            for i in range(15):
                fpr, tpr, thresholds = roc_curve(
                    all_labels[:, i], all_predictions[:, i]
                )
                auc_score = auc(fpr, tpr)
                auc_scores.append(auc_score)
            auc_score_avg = np.mean(auc_scores)

            if args.log_wandb:
                wandb.log(
                    {"evaluation_avg_loss": loss_avg, "auc_score_avg": auc_score_avg}
                )
                wandb.log(
                    {"per-label auc": dict(zip(FolderDataset.labels_list, auc_scores))}
                )

            logging.info(f"Average loss: {loss_avg}")
            logging.info(f"AUC score: {auc_score_avg}")

    if args.save_model:
        torch.save(model.state_dict(), "med-mae_hiera_tiny_224_classification.pth")


def init_args():
    parser = argparse.ArgumentParser(
        description="Train a model with a customizable learning rate."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for model training",
    )
    parser.add_argument(
        "--save_model", type=lambda x: bool(strtobool(x)), default=False
    )
    parser.add_argument("--log_wandb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0)  # 1e-8
    parser.add_argument(
        "--use_augmentations", type=lambda x: bool(strtobool(x)), default=False
    )
    parser.add_argument(
        "--rotation_angle", type=float, default=0.0
    )
    parser.add_argument("--head_dropout", type=float, default=0)  # 0.5
    parser.add_argument("--size", type=str, default="tiny")  # 0.5
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    main(args)
