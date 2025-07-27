from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from vacation.data import GalaxyDataset
from vacation.model.random_forest import hog_features


@click.group("rf", help="Commands related to the Random Forest")
def command():
    pass


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


command.add_command(hog)
