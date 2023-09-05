import logging
import wandb
from init_mae_sweep import mae_training_wrapper


if __name__ == "__main__":
    wandb.agent(
        sweep_id="sfch775i",
        project="mae-sweep",
        entity="med-hiera",
        function=mae_training_wrapper,
        count=1,
    )
