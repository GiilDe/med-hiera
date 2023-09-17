# Medical Hiera

 
Gilad Deutch,
Eran Levin


## Installation
```bash
pip install requirements.txt
```

## Using

### Data
In order to recreate this experiment you first to build the datasets "cocktail"

### MAE

```python
python mae_training.py --save_model_name "some model name.pth"...
```

### Classification

```python
python classification_training.py --pretrained_path "some model name.pth" --save_model_name "some other model name.pth"...
```


### Test

```python
python test_set_evaluation.py --pretrained_path "some other model name.pth"  ...
```

Note that this repo works with wandb, a link should be outputted in the start of each run (mae/classification/test), click to view run metrics.
