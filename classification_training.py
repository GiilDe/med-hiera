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
from sklearn.metrics import roc_curve, auc
import numpy as np

train_paths = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/train/"
test_paths = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/"
labels_path = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification/checxpert_data/Data_Entry_2017.csv"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def main(args):
    if args.log_wandb:
        wandb.login()
        wandb.init(
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
            },
        )

    model: Hiera = hiera_tiny_224(
        pretrained=False, checkpoint="hiera_in1k", num_classes=15
    )
    if ".pth" in args.pretrained_path:
        model_state_dict = torch.load(args.pretrained_path)
        logging.info(f"Loaded model from path {args.pretrained_path}")
    else:
        torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
        url = "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth"
        model_state_dict = torch.hub.load_state_dict_from_url(url)
        logging.info(f"Loaded model from url {url}")

    model.load_state_dict(model_state_dict, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_train = FolderDataset(
        paths=train_paths,
        labels_path=labels_path,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    dataset_test = FolderDataset(
        paths=test_paths,
        labels_path=labels_path,
    )
    logging.info(f"Train dataset size: {len(dataset_train)}")
    logging.info(f"Test dataset size: {len(dataset_test)}")
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    sigmoid = Sigmoid()
    loss_func = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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

            loss_avg /= len(dataloader_train)

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
            logging.info(f"Average loss: {loss_avg}")
            logging.info(f"AUC score: {auc_score_avg}")

    if args.save_model:
        torch.save(model.state_dict(), "med-mae_hiera_tiny_224_classification.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a customizable learning rate."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for model training",
    )
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
