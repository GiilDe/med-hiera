import wandb
from init_classification_sweep import classification_training_wrapper


if __name__ == "__main__":
    wandb.agent(
        sweep_id="fg2qtj5k",
        project="classification-sweep",
        entity="med-hiera",
        function=classification_training_wrapper,
        count=1,
    )
