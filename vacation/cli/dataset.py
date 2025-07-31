from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from vacation.data import (
    GalaxyDataset,
    augment_dataset,
    generate_dataset,
    train_test_split,
)
from vacation.data.augmentation import _transform


@click.group("dataset", help="Commands related to the dataset")
def command():
    pass


@click.command(
    "create",
    help="Download the original Galaxy10 DECaLS dataset and perform pre-processing and augmentation.",
)
@click.argument(
    "output_directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--seeds",
    type=int,
    nargs=3,
    default=(42, 42, 1337),
    help="The seeds (nargs=3) for (1) train-test split, (2) augmentation and (3) train-valid split.",
)
@click.option(
    "--pack-sizes",
    type=int,
    nargs=2,
    default=(1000, 1000),
    help="Sizes (nargs=2) of the data batches to process at the same time while splitting the data. "
    "Larger values increase memory consumption but decrease time consumption.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite previously created h5 files.",
)
@click.option(
    "--redownload",
    is_flag=True,
    help="Whether redownload the original dataset file from Zenodo.",
)
def create(
    output_directory: Path,
    seeds: tuple[int],
    pack_sizes: tuple[int],
    overwrite: bool,
    redownload: bool,
):

    # Download dataset and generate pre-processed ()
    try:
        generate_dataset(output_directory, overwrite=overwrite, redownload=redownload)
    except FileExistsError:
        pass

    # Train-Test split
    train_test_split(
        path=output_directory / "Galaxy10_DECals_proc.h5",
        random_state=seeds[0],
        pack_size=pack_sizes[0],
        overwrite=overwrite,
    )

    # Augmenting training data
    augment_dataset(
        path=output_directory / "Galaxy10_DECals_proc_train.h5",
        target_path=output_directory / "Galaxy10_DECals_augmented_train.h5",
        seed=seeds[1],
        overwrite=overwrite,
    )

    # Train-Validation split
    train_test_split(
        path=output_directory / "Galaxy10_DECals_augmented_train.h5",
        name_prefix="Galaxy10_DECals",
        test_type="valid",
        random_state=seeds[2],
        pack_size=pack_sizes[1],
        overwrite=overwrite,
    )


@click.command("plot", help="Create plots for a h5 dataset at a specific PATH.")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(
        ["all", "distribution", "examples", "augmented"], case_sensitive=True
    ),
    default="all",
    help="The type of the plot.",
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
    "--seed",
    "-s",
    type=int,
    default=None,
    help="The seed for the selection of examples.",
)
def plot(path: str, type: str, out: Path, seed: int | None):
    dataset = GalaxyDataset(path=path, device="cpu", cache_loaded=False)
    out.mkdir(exist_ok=True)

    if type == "all" or type == "distribution":
        dataset.plot_distribution(save_path=out / "distribution.pdf")
        print(f"Created {str(out / 'distribution.pdf')}")
    if type == "all" or type == "examples":
        dataset.plot_examples(
            random_state=seed, save_path=out / "examples.pdf", show_percentages=True
        )
        print(f"Created {str(out / 'examples.pdf')}")
    if type == "all" or type == "augmented":
        idx = int(np.random.default_rng(seed=seed).integers(0, len(dataset)))
        image = dataset[idx][0]

        fig, ax = plt.subplots(1, 2, layout="constrained")

        ax[0].imshow(image.swapaxes(2, 0))
        ax[0].set_title("Original Image")

        ax[1].imshow(_transform(image).swapaxes(2, 0))
        ax[1].set_title("Augmented Image")

        fig.savefig(str(out / "augmented.pdf"), bbox_inches="tight")
        print(f"Created {str(out / 'augmented.pdf')}")


command.add_command(create)
command.add_command(plot)
