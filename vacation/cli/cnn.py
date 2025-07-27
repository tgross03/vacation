from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import classification_report

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
    "--device",
    type=str,
    default=None,
    help="The device to load the model and dataset to. If None, the model's native device will be used.",
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
    device: str | None,
    seed: int | None,
    out: Path,
):

    if device is None:
        model = VCNN.load(model_path)
    else:
        model = VCNN.load(model_path, device=device)

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
    "--dataset",
    "-d",
    type=click.Choice(["both", "train", "valid"], case_sensitive=True),
    default="both",
    help="The dataset to plot the metric for.",
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
def plot_metric(path: str, type: str, dataset: str, out: Path):

    model = VCNN.load(path=path)

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
