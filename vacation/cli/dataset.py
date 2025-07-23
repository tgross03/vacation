from pathlib import Path

import click

from vacation.data import GalaxyDataset


@click.group("dataset", help="Commands related to the dataset")
def command():
    pass


@click.command("plot", help="Create plots for a h5 dataset at a specific PATH.")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(["all", "distribution", "examples"], case_sensitive=True),
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
        dataset.plot_examples(random_state=seed, save_path=out / "examples.pdf")
        print(f"Created {str(out / 'examples.pdf')}")


command.add_command(plot)
