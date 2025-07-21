from pathlib import Path

import optuna
from optuna.artifacts import FileSystemArtifactStore, download_artifact

from vacation.model import VCNN


def get_best_trial(
    storage_path: str, study_name: str, return_study: bool = False
) -> optuna.trial.FrozenTrial:

    storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    if return_study:
        return study.best_trial, study
    else:
        return study.best_trial


def get_model_from_trial(
    trial: optuna.trial.FrozenTrial,
    download_path: str,
    checkpoint_dir: str,
    overwrite: bool = False,
) -> VCNN:

    artifact_store = FileSystemArtifactStore(base_path=checkpoint_dir)

    if overwrite:
        Path(download_path).unlink(missing_ok=True)

    download_artifact(
        artifact_store=artifact_store,
        file_path=download_path,
        artifact_id=trial.user_attrs["artifact_id"],
    )

    model = VCNN.load(path=download_path)
    return model
