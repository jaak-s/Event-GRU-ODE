# Event prediction with GRU-ODE

This package allows training models that predict events in irregularly observed time series.

# Installation
In development mode the package can be installed as follows:
```
pip install -e .
```

# Example: toy data set
Training a model with the example toy data and no covariates.
The training uses total 10 epochs, and learning rate is decreased by 0.3 after 5 epochs.

```
cd Data/toy

python train.py \
  --data toy_10k.csv \
  --model gru_ode \
  --folds toy_10k_folds.csv \
  --fold_va 0 \
  --fold_te 1 \
  --dt 0.1 \
  --batch_size 64 \
  --hidden_size 64 \
  --epochs 10 \
  --lr_steps 5 \
  --output_hidden_size 64 \
  --event_periods 1.0 2.0 3.0 4.0 6.0 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --device cuda:0
```

Static (time independent) covariates can be specified by adding `--covs cov_data.csv`.

# Example with pickled sparse data
```
cd Data/toy

python train.py \
  --sparse_data toy_10k.pickle \
  --folds toy_10k_folds.csv \
  --fold_va 0 \
  --fold_te 1 \
  --dt 0.1 \
  --batch_size 64 \
  --hidden_size 64 \
  --epochs 2 \
  --lr_steps 5 \
  --output_hidden_size 64 \
  --event_periods 1.0 2.0 3.0 4.0 6.0 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --device cuda:0
```

# Saving models and predictions
Result performanced are always saved, but you can also save model and predictions:
* Use `--save_model 1` to save model at the end of the run.
* Use `--save_yhat 1` to save predictions on validation set, together with labels and columns.

## Results
The metrics and hyperparameters (settings) are stored in a .json file, which can be loaded:
```
import event_gruode as eg
res = eg.load_results("models/run_h64-64-64_do0.00-0.00_epochs2_lr0.001_lrsteps5_wd1.0e-05_dt0.1_ep1.0-2.0-3.0-4.0-6.0_fold_va0_fold_te1.json")

## check validation performance:
res["results_va"]["metrics"].T

## settings:
res["conf"]
```

## Loading saved yhat
The yhat values (predictions) can be loaded by
```
import pandas as pd
import torch

pred = torch.load("models/run_h64-64-64_do0.00...-yhat.pt")

## yhat (probabilities, labels etc)
df = pd.DataFrame(pred)
df.head()
```

# Example with Optuna with Hyperband
Here we use the defaults for all variables except `hidden_size` which we set a limit from 64 to 256 with 64 step.
The last line `--optuna_hyperband 1` switches on Hyperband pruner with total of 5 trials.
```
python train_optuna.py \
  --data toy_10k.csv \
  --model gru_ode \
  --ode_slower 10 \
  --folds toy_10k_folds.csv \
  --fold_va 0 \
  --fold_te 1 \
  --dt 0.1 \
  --batch_size 64 \
  --hidden_size_low 64 \
  --hidden_size_high 256 \
  --hidden_size_step 64 \
  --epochs 10 \
  --lr_steps 8 \
  --event_periods 1.0 2.0 3.0 4.0 6.0 \
  --device cuda:0 \
  --optuna_n_trials 5 \
  --optuna_hyperband 1 \
  --optuna_storage sqlite:///hp_scans.sqlite \
  --optuna_study_name scan1
```
