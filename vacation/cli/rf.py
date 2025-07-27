from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from vacation.data import GalaxyDataset
from vacation.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_example_matrix,
)
from vacation.model.random_forest import hog_features


@click.group("rf", help="Commands related to the Random Forest")
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
    help="The file to load the features from. If no file with this name is know,"
    " the features will be created. If None, the features won't be loaded.",
)
@click.option(
    "--save-to",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    default=None,
    help="The directory to save the features to. If None, the features won't be saved.",
)
@click.option(
    "--pre-process",
    is_flag=True,
    help="Whether to apply pre-processing to the images before the evaluation.",
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
    pre_process: bool,
    seed: int | None,
    out: Path,
):

    model = joblib.load(model_path)

    dataset = GalaxyDataset(path=dataset_path, device="cpu", cache_loaded=False)

    if load_from is not None and load_from.exists():
        features = np.load(load_from)
    else:
        features, _, _ = hog_features(dataset, augmented=pre_process)

    if save_to is not None:
        np.save(save_to / f"rf_features{'_proc' if pre_process else ''}.npy", features)
        print(
            f"Saved {len(features)} features to {str(save_to / f'rf_features{'_proc' if pre_process else ''}.npy')}."
        )

    y_pred = torch.from_numpy(model.predict(features))
    y_true = dataset.get_labels()

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
        save_path=out / f"rf_example_matrix{'_proc' if pre_process else ''}.pdf",
        figsize=(5, 5),
        seed=seed,
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=True,
        save_path=out / f"rf_confusion_matrix{'_proc' if pre_process else ''}.pdf",
    )


@click.command(
    "hog",
    help="Create an example plot of a Histogram of Oriented Gradients from an image from a dataset at a specific PATH.",
)
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
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
@click.option(
    "--index",
    "-i",
    type=int,
    default=None,
    help="The index of the example.",
)
def hog(path: str, out: Path, index: int | None):
    dataset = GalaxyDataset(path=path, device="cpu", cache_loaded=False)
    out.mkdir(exist_ok=True)

    idx = (
        int(np.random.default_rng(seed=None).integers(0, len(dataset)))
        if index is None
        else index
    )

    fig, ax = plt.subplots(1, 2, figsize=(7, 5), layout="constrained")

    ax[0].imshow(dataset[idx][0].swapaxes(2, 0))
    ax[0].set_title("Original Image")

    ax[1].imshow(
        hog_features(
            df=dataset, visualize_only=True, sample_img_idx=idx, augmented=True
        ).T,
        cmap="inferno",
    )
    ax[1].set_title("Histogram of Oriented Gradients")

    fig.savefig(str(out / "hog_example.pdf"), bbox_inches="tight")
    print(f"Created {str(out / 'hog_example.pdf')}")


command.add_command(evaluate)
command.add_command(hog)
