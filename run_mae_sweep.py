import logging
import wandb
from mae_training import main, init_args
from init_mae_sweep import mae_training_wrapper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


if __name__ == "__main__":
    wandb.agent(
        sweep_id="aofl54ju",
        project="mae-sweep",
        entity="med-hiera",
        function=mae_training_wrapper,
        count=1,
    )
