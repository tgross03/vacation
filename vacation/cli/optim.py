import tempfile
from pathlib import Path

import click
import joblib
import numpy as np
import optuna
import torch
from optuna.artifacts import FileSystemArtifactStore, download_artifact, upload_artifact
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from vacation.data import GalaxyDataset
from vacation.evaluation.optimization_results import get_best_trial
from vacation.evaluation.visualizations import plot_hyperparameter_importance
from vacation.model import VCNN
from vacation.model.random_forest import hog_features


@click.group("optim", help="Commands related to the Hyperparameter Optimization.")
def command():
    pass


@click.command(
    "cnn",
    help="Starts the hyperparameter optimization for the CNN using optuna. (Requires GPU!)",
)
@click.argument(
    "train_dataset",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "valid_dataset",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="The directory where the checkpoints of the optimization aka. the artifacts should be created.",
)
@click.option(
    "--num-trials",
    type=int,
    default=300,
    help="The number of trials to run.",
)
@click.option(
    "--num-epochs",
    type=int,
    default=100,
    help="The number of epochs to run at maximum per trial.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the training on. Has to be a pytorch supported GPU device (e.g. cuda)!",
)
@click.option(
    "--train-cache-size",
    type=str,
    default="9G",
    help="Maximum cache size of the training data.",
)
@click.option(
    "--valid-cache-size",
    type=str,
    default="3G",
    help="Maximum cache size of the validation data.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="The seed of the CNN (2).",
)
def cnn(
    train_dataset: Path,
    valid_dataset: Path,
    checkpoint_dir: Path,
    num_trials: int,
    num_epochs: int,
    device: str,
    train_cache_size: str,
    valid_cache_size: str,
    seed: int,
):
    """
    This command can be used to start a hyperparameter optimization using optuna.
    It is partially based on the pytorch example script provided by optuna:
    https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_checkpoint.py
    """

    N_TRIALS = num_trials
    N_EPOCHS = num_epochs
    CHECKPOINT_DIR = checkpoint_dir
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

    artifact_store = FileSystemArtifactStore(base_path=CHECKPOINT_DIR)

    # Initialize Datasets to avoid reloading data from disk
    train_ds = GalaxyDataset(
        path=str(train_dataset),
        device=device,
        max_cache_size=train_cache_size,
        cache_loaded=True,
    )

    valid_ds = GalaxyDataset(
        path=str(valid_dataset),
        device=device,
        max_cache_size=valid_cache_size,
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
            "num_conv_blocks": int(
                trial.suggest_int(name="num_conv_blocks", low=2, high=6)
            ),
            "out_channels": [],
            "conv_dropout_rates": [],
            "num_dense_layers": int(
                trial.suggest_int(name="num_dense_layers", low=1, high=3)
            ),
            "lin_out_features": [],
            "lin_dropout_rates": [],
            "optimizer": getattr(
                torch.optim,
                trial.suggest_categorical(
                    "optimizer", ["Adam", "AdamW", "NAdam", "SGD"]
                ),
            ),
            "activation_func": getattr(
                torch.nn,
                trial.suggest_categorical(
                    "activation_func", ["PReLU", "ReLU", "LeakyReLU"]
                ),
            ),
            "learning_rate": trial.suggest_float(
                name="learning_rate", low=1e-4, high=1e-2
            ),
            "weight_decay": trial.suggest_float(
                name="weight_decay", low=1e-3, high=1e-1
            ),
        }

        # Suggest values for out_channels and dropout_rates for Conv Blocks
        for i in range(0, hyper_params["num_conv_blocks"]):
            # low = 1 if i == 0 else hyper_params["out_channels"][i - 1]
            hyper_params["out_channels"].append(
                int(trial.suggest_int(name=f"out_channels_{i}", low=1, high=12))
            )
            hyper_params["conv_dropout_rates"].append(
                trial.suggest_float(name=f"conv_dropout_rate_{i}", low=0.0, high=1.0)
            )

        # Suggest values for lin_out_features and dropout_rates for dense layers
        for i in range(0, hyper_params["num_dense_layers"]):
            high = 1000 if i == 0 else hyper_params["lin_out_features"][i - 1]
            hyper_params["lin_out_features"].append(
                int(trial.suggest_int(name=f"lin_out_features_{i}", low=50, high=high))
            )
            hyper_params["lin_dropout_rates"].append(
                trial.suggest_float(name=f"lin_dropout_rate_{i}", low=0.0, high=1.0)
            )

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
                    device=device,
                    seed=seed,
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

    storage = optuna.storages.RDBStorage(
        "sqlite:///vacation.sqlite3",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )

    study = optuna.create_study(
        study_name="vacation_v2",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

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


@click.command(
    "rf",
    help="Starts the hyperparameter optimization for the CNN using optuna. (Requires GPU!)",
)
@click.argument(
    "train_dataset",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "valid_dataset",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "training_dataset_non_augmented",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--augmented",
    type=bool,
    default=True,
    help="Whether to train on the augmented datasets or the original one.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="The directory where the checkpoints of the optimization aka. the artifacts should be created.",
)
@click.option(
    "--n-iter",
    type=int,
    default=75,
    help="The number of configurations to draw.",
)
@click.option(
    "--k-fold", type=int, default=5, help="The fold for the cross-validation."
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the training on. Has to be a pytorch supported GPU device (e.g. cuda)!",
)
@click.option(
    "--train-cache-size",
    type=str,
    default="9G",
    help="Maximum cache size of the training data.",
)
@click.option(
    "--valid-cache-size",
    type=str,
    default="3G",
    help="Maximum cache size of the validation data.",
)
@click.option("--seed", type=int, default=42, help="The seed of the optimization.")
def rf(
    train_dataset: Path,
    valid_dataset: Path,
    training_dataset_non_augmented: Path,
    augmented: bool,
    checkpoint_dir: Path,
    n_iter: int,
    k_fold: int,
    device: str,
    train_cache_size: str,
    valid_cache_size: str,
    seed: int,
):

    # initialize augmented datasets
    train_ds_aug = GalaxyDataset(
        path=str(train_dataset),
        device=device,
        max_cache_size=train_cache_size,
        cache_loaded=True,
    )

    valid_ds_aug = GalaxyDataset(
        path=str(valid_dataset),
        device=device,
        max_cache_size=valid_cache_size,
        cache_loaded=True,
    )

    # initialize original dataset
    train_ds = GalaxyDataset(
        path=str(training_dataset_non_augmented),
        device=device,
        max_cache_size=train_cache_size,
        cache_loaded=True,
    )

    # augmented datasets already split in train and validation

    location = checkpoint_dir
    num_npy_files = len([file for file in location.glob("*.npy") if file.is_file()])

    if num_npy_files < 4:
        print("Extracting HOG features from augmented datasets...")
        X_train_aug, y_train_aug, sample_image = hog_features(train_ds_aug)
        X_val_aug, y_val_aug, _ = hog_features(valid_ds_aug)

        # original dataset
        print("Extracting HOG features from original dataset...")
        X_train, y_train, _ = hog_features(train_ds, augmented=False)

        # saving features and labels
        np.save(str(location / "rf_features_train_aug.npy"), X_train_aug)
        np.save(str(location / "rf_labels_train_aug.npy"), y_train_aug)
        np.save(str(location / "rf_features_valid_aug.npy"), X_val_aug)
        np.save(str(location / "rf_labels_valid_aug.npy"), y_val_aug)
        np.save(str(location / "rf_features_train.npy"), X_train)
        np.save(str(location / "rf_labels_train.npy"), y_train)

    else:
        # loading if not features already saved
        X_train_aug = np.load(str(location / "rf_features_train_aug.npy"))
        y_train_aug = np.load(str(location / "rf_labels_train_aug.npy"))
        X_val_aug = np.load(str(location / "rf_features_valid_aug.npy"))
        y_val_aug = np.load(str(location / "rf_labels_valid_aug.npy"))
        X_train = np.load(str(location / "rf_features_train.npy"))
        y_train = np.load(str(location / "rf_labels_train.npy"))

    # hyperparameter tuning for Random Forest
    print("Starting hyperparameter optimization for Random Forest...")
    RF = RandomForestClassifier(random_state=42)

    # parameter distribution for optimization
    param_dist = {
        "n_estimators": randint(50, 400),
        "max_features": ["sqrt", "log2", None],
        "max_depth": randint(1, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 2),
    }

    rand_search = RandomizedSearchCV(
        RF,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=k_fold,
        n_jobs=-1,
        verbose=10,
        random_state=seed,
    )
    if augmented:
        rand_search.fit(X_train_aug, y_train_aug)
    else:
        rand_search.fit(X_train, y_train)

    best_rf = rand_search.best_estimator_
    print("Best hyperparameters are:", rand_search.best_params_)

    filename = str(location / "rf_optimized.sav")
    joblib.dump(best_rf, filename)
    print(f"Saved best model to {filename}.")


@click.command(
    "plot_importance", help="Plots the Hyperparameter importance of a Optuna study."
)
@click.argument(
    "storage_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "study_name",
    type=str,
)
@click.option(
    "--out",
    "-o",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    default=Path("./").absolute(),
    help="The output directory to create the plots at.",
)
def plot_importance(
    storage_path: Path,
    study_name: str,
    out: Path,
):

    _, study = get_best_trial(
        storage_path=str(storage_path),
        study_name=study_name,
        return_study=True,
    )

    plot_hyperparameter_importance(
        study=study,
        save_path=out / "cnn_hyperparameter_importance.pdf",
        log=False,
    )


command.add_command(cnn)
command.add_command(rf)
command.add_command(plot_importance)
