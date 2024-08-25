import event_gruode as eg
import optuna
import argparse
import numpy as np
import os.path

def objective(conf, trial):
    """Setting up parameters"""
    conf.hidden_size = trial.suggest_int("hidden_size",
            low  = conf.hidden_size_low,
            high = conf.hidden_size_high,
            step = conf.hidden_size_step)

    conf.output_hidden_size = trial.suggest_int("output_hidden_size",
            low  = conf.output_hidden_size_low,
            high = conf.output_hidden_size_high,
            step = conf.output_hidden_size_step)

    conf.cov_hidden_size = trial.suggest_int("cov_hidden_size",
            low  = conf.cov_hidden_size_low,
            high = conf.cov_hidden_size_high,
            step = conf.cov_hidden_size_step)

    conf.output_dropout = trial.suggest_float("output_dropout",
            low  = conf.output_dropout_low,
            high = conf.output_dropout_high,
            step = conf.output_dropout_step)

    conf.cov_dropout = trial.suggest_float("cov_dropout",
            low  = conf.cov_dropout_low,
            high = conf.cov_dropout_high,
            step = conf.cov_dropout_step)

    conf.weight_decay = trial.suggest_loguniform("weight_decay",
            low  = conf.weight_decay_low,
            high = conf.weight_decay_high)

    run = eg.create_run(conf, trial=trial)
    score = run.train()
    return score

def parse_and_run():
    parser = argparse.ArgumentParser(description="Testing EventGRUBayes")
    parser.add_argument("--sparse_data", help="Data file for sparse data (.pickle)", type=str, default=None)
    parser.add_argument("--data", help="Data file (.cvs)", type=str, default=None)
    parser.add_argument("--covs", help="Covariates file (.csv)", type=str, default=None)
    parser.add_argument("--folds", help="Folding file (.csv)", type=str, default=None, required=True)
    parser.add_argument("--fold_va", help="Validation fold (default 0)", type=int, default=0)
    parser.add_argument("--fold_te", help="Test fold (default 1)", type=int, default=1)
    parser.add_argument("--dt", help="Time step (default 0.1)", type=float, default=0.1)
    parser.add_argument("--ode_slower", help="ODE scale, 10 means ODE step is (dt / 10)", type=float, default=10.0)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--model", help="Batch size", type=str,
        default = "gru_discrete",
        choices = ["gru_discrete", "gru_ode"],
    )
    parser.add_argument("--hidden_size_low", help="Hidden size low", type=int, default=128)
    parser.add_argument("--hidden_size_high", help="Hidden size high", type=int, default=256)
    parser.add_argument("--hidden_size_step", help="Hidden size step", type=int, default=32)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--output_hidden_size_low", help="Out hidden size low", type=int, default=128)
    parser.add_argument("--output_hidden_size_high", help="Out hidden size high", type=int, default=256)
    parser.add_argument("--output_hidden_size_step", help="Out hidden size step", type=int, default=32)
    parser.add_argument("--cov_hidden_size_low", help="Cov hidden size low", type=int, default=128)
    parser.add_argument("--cov_hidden_size_high", help="Cov hidden size high", type=int, default=256)
    parser.add_argument("--cov_hidden_size_step", help="Cov hidden size step", type=int, default=32)
    parser.add_argument("--output_dropout_low", help="Output dropout low", type=float, default=0.0)
    parser.add_argument("--output_dropout_high", help="Output dropout high", type=float, default=0.2)
    parser.add_argument("--output_dropout_step", help="Output dropout step", type=float, default=0.05)
    parser.add_argument("--cov_dropout_low", help="Cov dropout low", type=float, default=0.0)
    parser.add_argument("--cov_dropout_high", help="Cov dropout high", type=float, default=0.2)
    parser.add_argument("--cov_dropout_step", help="Cov dropout step", type=float, default=0.05)
    parser.add_argument("--event_periods", help="Event periods to predict (into future)", nargs="+", default=[1.0, 2.0, 3.0, 4.0, 6.0], type=float)
    parser.add_argument("--add_events_to_x", help="Whether to add events to x matrix", type=int, default=0)
    parser.add_argument("--noevent_pad", help="Padding for final observation", type=float, default=0.0)
    parser.add_argument("--label_buffer", help="Assumes patients are fine for buffer duration (reduces NAs at the end of traj.)", type=float, default=30.0)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--lr_steps", help="Epoch numbers when to drop lr", type=int, nargs="+", default=[40])
    parser.add_argument("--lr_alpha", help="Drop multiplier for lr", type=float, default=0.3)
    parser.add_argument("--weight_decay_low", help="Weight decay low", type=float, default=1e-6)
    parser.add_argument("--weight_decay_high", help="Weight decay high", type=float, default=1e-4)
    parser.add_argument("--device", help="Device", type=str, default="cuda:0")
    parser.add_argument("--num_data_workers", help="Number of data workers", type=int, default=0)
    parser.add_argument("--save_board", help="Whether to save TensorBoard", type=int, default=1)
    parser.add_argument("--save_model", help="Whether to save model", type=int, default=0)
    parser.add_argument("--save_yhat", help="Whether to save trajectories (yhat, label)", type=int, default=0)
    parser.add_argument("--board_dir", help="Path where to save TensorBoard", type=str, default="boards/")
    parser.add_argument("--output_dir", help="Path where to save conf and model", type=str, default="models/")
    parser.add_argument("--early_stopping_periods", help="Early stopping periods (0 means no stopping)", type=int, default=0)
    parser.add_argument("--optuna_hyperband", help="Whether to use hyperband", type=int, default=0)
    parser.add_argument("--optuna_n_trials", help="Number of trials to use", type=int, default=50)
    parser.add_argument("--optuna_storage", help="Where the study is stored (or leaded if exists)", type=str, default="sqlite:///optuna.sqlite")
    parser.add_argument("--optuna_study_name", help="Optuna study name", type=str, default="default")
    parser.add_argument("--limit_train_batches", type=float, help="Use only subset of batches for training", default=1.0)
    parser.add_argument("--limit_val_batches", type=float, help="Use only subset of batches for validation", default=1.0)
    parser.add_argument("--limit_test_batches", type=float, help="Use only subset of batches for test", default=1.0)

    conf = parser.parse_args()
    print(conf)

    if conf.optuna_hyperband:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource     = 1,
            max_resource     = conf.epochs,
            reduction_factor = 3,
        )
    else:
        pruner = None

    print(f"Executing Optuna to scan {conf.optuna_n_trials} trials.")
    ## switch to use sqlite
    print(f"Optuna storage:    '{conf.optuna_storage}'")
    print(f"Optuna study_name: '{conf.optuna_study_name}'")
    study = optuna.create_study(direction='maximize', pruner=pruner, storage=conf.optuna_storage, load_if_exists=True, study_name=conf.optuna_study_name)

    study.optimize(lambda trial: objective(conf, trial), n_trials=conf.optuna_n_trials)
    print("You can load study by:")
    print(f"""  study = optuna.load_study(storage="{conf.optuna_storage}", study_name="{conf.optuna_study_name}")""")

    print()
    print("-----[ Optuna results ]-----")
    print(f"Best value:  {study.best_value}")
    print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    parse_and_run()
