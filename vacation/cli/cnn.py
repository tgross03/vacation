from pathlib import Path

import click

from vacation.model import VCNN


@click.group("cnn", help="Commands related to the Convolutional Neural Network.")
def command():
    pass


@click.command("plot_metric")
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


command.add_command(plot_metric)
