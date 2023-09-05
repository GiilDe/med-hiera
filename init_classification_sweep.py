import logging
import wandb
from classification_training import main, init_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def classification_training_wrapper():
    run = wandb.init()
    default_args = init_args()
    default_args.weight_decay = wandb.config.weight_decay
    default_args.learning_rate = wandb.config.learning_rate
    default_args.use_augmentations = wandb.config.use_augmentations
    default_args.epochs = wandb.config.epochs
    default_args.head_dropout = wandb.config.head_dropout
    default_args.log_wandb = True
    default_args.pretrained_path = ""
    if default_args.pretrained_path == "":
        raise ValueError("pretrained_path must be set")
    main(default_args)


if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "auc_score_avg"},
        "parameters": {
            "weight_decay": {"max": 0.1, "min": 0.0},
            "learning_rate": {"max": 5e-4, "min": 1e-5},
            "use_augmentations": {"values": [True, False]},
            "epochs": {"values": [10, 20, 30, 40, 50]},
            "head_dropout": {"max": 0.5, "min": 0.0},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="classification-sweep", entity="med-hiera"
    )
    logging.info(f"Sweep ID: {sweep_id}")
