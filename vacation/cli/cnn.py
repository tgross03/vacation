from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from vacation.data import GalaxyDataset
from vacation.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_example_matrix,
)
from vacation.model import VCNN


@click.group("cnn", help="Commands related to the Convolutional Neural Network.")
def command():
    pass


@click.command("eval", help="Evaluates a saved model's performance on a given dataset.")
@click.argument(
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--load-from",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    default=None,
    help="The file to load the evaluation results from. If no file with this name is know, "
    "the evaluation will be created. If None, the results won't be loaded.",
)
@click.option(
    "--save-to",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    default=None,
    help="The directory to save the evaluation results to. If None, the results won't be saved.",
)
@click.option(
    "--dataset-directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    default=None,
    help="The directory where the Galaxy10_DECaLS_train.h5 and Galaxy10_DECaLS_valid.h5 dataset are saved."
    "If None, the datasets given in the .pt file of the model will be used.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="The device to load the model and dataset to. If None, the cpu will be used.",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="The seed for the selection of examples.",
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
def evaluate(
    model_path: str,
    dataset_path: str,
    load_from: Path | None,
    save_to: Path | None,
    dataset_directory: Path | None,
    device: str | None,
    seed: int | None,
    out: Path,
):

    if dataset_directory is not None:
        train_ds = GalaxyDataset(
            dataset_directory / "Galaxy10_DECaLS_train.h5",
            device=device,
            cache_loaded=False,
        )
        valid_ds = GalaxyDataset(
            dataset_directory / "Galaxy10_DECaLS_valid.h5",
            device=device,
            cache_loaded=False,
        )
    else:
        train_ds, valid_ds = None, None

    model = VCNN.load(
        model_path, device=device, train_dataset=train_ds, valid_dataset=valid_ds
    )

    dataset = GalaxyDataset(path=dataset_path, device=model._device)

    if load_from is None or not load_from.exists():
        y_pred, y_true = model.predict_dataset(dataset=dataset, return_true=True)

        if save_to is not None:
            np.save(save_to / "cnn_evaluation_results.npy", y_pred.cpu().numpy())
            print(
                f"Saved {len(y_pred)} evaluation results to {str(save_to / 'cnn_evaluation_results.npy')}."
            )

    else:
        y_pred = torch.from_numpy(np.load(load_from)).to(model._device)
        y_true = dataset.get_labels()
        print(
            f"Loaded {len(y_pred)} evaluation results from {str(save_to / 'cnn_evaluation_results.npy')}."
        )

    print(
        classification_report(y_pred=y_pred.cpu().numpy(), y_true=y_true.cpu().numpy())
    )
    print(
        "Accuracy",
        accuracy_score(y_pred=y_pred.cpu().numpy(), y_true=y_true.cpu().numpy()),
    )

    plot_example_matrix(
        dataset=dataset,
        y_pred=y_pred,
        save_path=out / "cnn_example_matrix.pdf",
        layout=(4, 3),
        true_false_ratio=0.5,
        figsize=(5, 7),
        seed=seed,
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=True,
        save_path=out / "cnn_confusion_matrix.pdf",
    )


@click.command("plot_metric", help="Plots the metrics of a saved model.")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(["both", "loss", "accuracy"], case_sensitive=True),
    default="both",
    help="The metric to plot.",
)
@click.option(
    "--dataset-directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    default=None,
    help="The directory where the Galaxy10_DECaLS_train.h5 and Galaxy10_DECaLS_valid.h5 dataset are saved."
    "If None, the datasets given in the .pt file of the model will be used.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["both", "train", "valid"], case_sensitive=True),
    default="both",
    help="The dataset to plot the metric for.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="The device to load the model and dataset to. If not provided, cpu will be used.",
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
def plot_metric(
    path: str,
    type: str,
    dataset_directory: Path | None,
    dataset: str,
    device: str,
    out: Path,
):

    if dataset_directory is not None:
        train_ds = GalaxyDataset(
            dataset_directory / "Galaxy10_DECaLS_train.h5",
            device=device,
            cache_loaded=False,
        )
        valid_ds = GalaxyDataset(
            dataset_directory / "Galaxy10_DECaLS_valid.h5",
            device=device,
            cache_loaded=False,
        )
    else:
        train_ds, valid_ds = None, None

    model = VCNN.load(
        path=path, train_dataset=train_ds, valid_dataset=valid_ds, device=device
    )

    if dataset == "both":
        dataset = ["train", "valid"]
    else:
        dataset = [dataset]

    if type == "both" or type == "loss":
        model.plot_metric(
            key="loss", components=dataset, save_path=out / "cnn_loss.pdf"
        )
    if type == "both" or type == "accuracy":
        model.plot_metric(
            key="accuracy", components=dataset, save_path=out / "cnn_accuracy.pdf"
        )


command.add_command(evaluate)
command.add_command(plot_metric)
