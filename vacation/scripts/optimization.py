import tempfile
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.artifacts import FileSystemArtifactStore, download_artifact, upload_artifact
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState

from vacation.data import GalaxyDataset
from vacation.model import VCNN

rng = np.random.default_rng(seed=1337)

"""
This script can be used to start a hyperparameter optimization using optuna.
It is based on the pytorch example script provided by optuna:

https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_checkpoint.py

"""

N_EPOCHS = 100
CHECKPOINT_DIR = Path("/scratch/tgross/vacation_models/artifacts")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

artifact_store = FileSystemArtifactStore(base_path=CHECKPOINT_DIR)

# Initialize Datasets to avoid reloading data from disk
train_ds = GalaxyDataset(
    path="/scratch/tgross/vacation_data/Galaxy10_DECals_train.h5",
    device="cuda:1",
    max_cache_size="15G",
    cache_loaded=True,
)

valid_ds = GalaxyDataset(
    path="/scratch/tgross/vacation_data/Galaxy10_DECals_valid.h5",
    device="cuda:1",
    max_cache_size="4G",
    cache_loaded=True,
)


def objective(trial: optuna.trial.Trial):

    # Define hyperparameters
    hyper_params = {
        "train_batch_size": int(
            2 ** (trial.suggest_int(name="train_batch_size", low=0, high=9))
        ),
        "valid_batch_size": int(
            2 ** (trial.suggest_int(name="valid_batch_size", low=0, high=9))
        ),
        "out_channels": [
            int(trial.suggest_int(name="out_channels_0", low=1, high=12)),
            int(trial.suggest_int(name="out_channels_1", low=1, high=12)),
        ],
        "dropout_rates": [
            trial.suggest_float(name="dropout_rates_0", low=0.0, high=0.75),
            trial.suggest_float(name="dropout_rates_1", low=0.0, high=0.75),
            trial.suggest_float(name="dropout_rates_2", low=0.0, high=0.75),
        ],
        "lin_out_features": [
            int(trial.suggest_int(name="lin_out_features_0", low=50, high=500))
        ],
        "optimizer": getattr(
            torch.optim,
            trial.suggest_categorical("optimizer", ["Adam", "AdamW", "NAdam", "SGD"]),
        ),
        "activation_func": getattr(
            torch.nn,
            trial.suggest_categorical(
                "activation_func", ["PReLU", "ReLU", "LeakyReLU"]
            ),
        ),
        "learning_rate": trial.suggest_float(name="weight_decay", low=1e-4, high=1e-2),
        "weight_decay": trial.suggest_float(name="learning_rate", low=1e-3, high=1e-1),
    }

    # Check if trial failed and has an artifact to jump back to
    artifact_id = None
    retry_history = RetryFailedTrialCallback.retry_history(trial)
    for trial_number in reversed(retry_history):
        artifact_id = trial.study.trials[trial_number].user_attrs.get("artifact_id")
        if artifact_id is not None:
            retry_trial_number = trial_number
            break

    with tempfile.TemporaryDirectory(dir=CHECKPOINT_DIR) as tempdir:
        checkpoint_path = Path(tempdir) / "model.pt"

        if artifact_id is not None:

            print(f"Loading checkpoint from trial {retry_trial_number}.")

            download_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                artifact_id=artifact_id,
            )

            model = VCNN.load(
                path=str(checkpoint_path),
                train_dataset=train_ds,
                valid_dataset=valid_ds,
            )

        else:

            # Init model
            model = VCNN(
                **hyper_params,
                loss_func=torch.nn.CrossEntropyLoss,
                device="cuda:1",
                seed=42,
            )

            # Init data
            model.init_data(train_dataset=train_ds, valid_dataset=valid_ds)

        model.train_epochs(
            n_epochs=N_EPOCHS - model._epoch,
            trial=trial,
            trial_metric="accuracy",
            save_path=str(checkpoint_path),
            save_interval=1,
        )

        artifact_id = upload_artifact(
            artifact_store=artifact_store,
            file_path=checkpoint_path,
            study_or_trial=trial,
        )

        trial.set_user_attr("artifact_id", artifact_id)

    return model._metrics["accuracy"].valid_vals[-1].cpu().numpy()


if __name__ == "__main__":

    storage = optuna.storages.RDBStorage(
        "sqlite:///vacation.sqlite3",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )

    study = optuna.create_study(
        study_name="vacation",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print(" -> Number of finished trials: ", len(study.trials))
    print(" -> Number of pruned trials: ", len(pruned_trials))
    print(" -> Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print(" -> Value: ", trial.value)
    print(" -> artifact_id: ", trial.user_attrs.get("artifact_id"))

    print(" -> Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
