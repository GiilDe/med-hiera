import logging
import wandb
from mae_training import main, init_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def mae_training_wrapper():
    run = wandb.init()
    default_args = init_args()
    default_args.weight_decay = wandb.config.weight_decay
    default_args.mask_ratio = wandb.config.mask_ratio
    default_args.learning_rate = wandb.config.learning_rate
    default_args.use_augmentations = wandb.config.use_augmentations
    default_args.epochs = wandb.config.epochs
    default_args.log_wandb = True
    main(default_args)


if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "evaluation_avg_loss"},
        "parameters": {
            # "weight_decay": {"max": 0.05, "min": 0.0},
            "weight_decay": {"value": 0.0},
            "mask_ratio": {"max": 0.8, "min": 0.6},
            "learning_rate": {"max": 5e-4, "min": 1e-5},
            "use_augmentations": {"values": [True, False]},
            "epochs": {"values": [10, 20, 30, 40, 50]},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="mae-sweep", entity="med-hiera"
    )
    logging.info(f"Sweep ID: {sweep_id}")
