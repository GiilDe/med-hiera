import torch
from hiera.hiera import Hiera, hiera_small_224, hiera_tiny_224
from hiera.hiera_mae import MaskedAutoencoderHiera
from utils import FolderDataset
import logging
from tqdm import tqdm
import argparse
from distutils.util import strtobool
from torch.nn import BCELoss, Sigmoid
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from torch.utils.data import DataLoader

test_paths = [
    "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification_processed/checxpert_data/test/",
]
labels_path = "/home/yandex/MLFH2023/giladd/hiera/datasets/datasets_classification/checxpert_data/Data_Entry_2017.csv"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def main(args):
    ### Load Model ###
    torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
    pretrained_hub = args.pretrained_path == ""
    if args.size == "tiny":
        model: Hiera = hiera_tiny_224(
            pretrained=False,
            checkpoint=None,
            num_classes=15,
        )
    else:
        model: Hiera = hiera_small_224(
            pretrained=False,
            checkpoint=None,
            num_classes=15,
        )
    if ".pth" in args.pretrained_path:
        model_state_dict = torch.load(args.pretrained_path)
        logging.info(f"Loaded model from path {args.pretrained_path}")
    else:
        torch.hub.set_dir("/home/yandex/MLFH2023/giladd/hiera/")
        url = "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth"
        model_state_dict = torch.hub.load_state_dict_from_url(url)["model_state"]
        logging.info(f"Loaded model from url {url}")
        if "head.projection.weight" in model_state_dict:
            del model_state_dict["head.projection.weight"]
        if "head.projection.bias" in model_state_dict:
            del model_state_dict["head.projection.bias"]
    logging.info(f"Model loaded from torch.hub: {pretrained_hub}")
    if args.pretrained_path != "":
        logging.info(f"Loading pretrained model from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path))
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
                "batch_size": args.batch_size,
                "size": args.size,
            },
        )

    dataset_test = FolderDataset(
        paths=test_paths,
        labels_path=labels_path,
    )
    logging.info(f"Train dataset size: {len(dataset_test)}")

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    sigmoid = Sigmoid()
    loss_func = BCELoss()
    ### Evaluate Model ###
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
            fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_predictions[:, i])
            auc_score = auc(fpr, tpr)
            auc_scores.append(auc_score)
        auc_score_avg = np.mean(auc_scores)

        if args.log_wandb:
            wandb.log({"evaluation_avg_loss": loss_avg, "auc_score_avg": auc_score_avg})
            wandb.log(
                {"per-label auc": dict(zip(FolderDataset.labels_list, auc_scores))}
            )

        logging.info(f"Average loss: {loss_avg}")
        logging.info(f"AUC score: {auc_score_avg}")


def init_args():
    parser = argparse.ArgumentParser(description="test a model")
    parser.add_argument("--log_wandb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--size", type=str, default="tiny")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    main(args)
