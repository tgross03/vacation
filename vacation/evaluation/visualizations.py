import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from vacation.data import CLASS_NAMES, GalaxyDataset


# Based on:
# https://stackoverflow.com/a/7944576
# https://stackoverflow.com/a/39598358
# https://stackoverflow.com/a/69099757
def _set_axes_border(ax: matplotlib.axes._axes.Axes, color: str, width: float):
    [x.set_linewidth(width) for x in ax.spines.values()]

    ax.set_xticks([])
    ax.set_yticks([])

    plt.setp(ax.spines.values(), color=color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)

    return ax


# The next two methods are taken from an own project:
# https://github.com/tgross03/simtools/blob/main/simtools/simulations/simulation_chain.py
def _plot_text(
    text: str,
    ax: matplotlib.axes._axes.Axes,
    pos: tuple[float] = (0, 1),
    text_options: dict = dict(fontsize=12, fontfamily="monospace"),
    bbox: dict = dict(edgecolor="black", boxstyle="round"),
):
    """

    Plot a boxed text onto a matplotlib plot

    Parameters
    ----------
    text : str
        The text to write in the box

    ax : matplotlib.axes._axes.Axes
        A axis to put the text into

    pos : tuple, optional
        The relative position of the text box

    text_options : dict, optional
        The options for the annotation

    bbox : dict, optional
        The bbox argument of the annotation parameters

    """

    textanchor = ax.get_window_extent()
    ax.annotate(
        text,
        pos,
        xycoords=textanchor,
        va="top",
        bbox=bbox,
        **text_options,
    )


def _plot_label(text: str, ax, facecolor: str):
    """

    Plot a boxed label in the upper left corner of the plot

    Parameters
    ----------
    text : str
        The text to write in the box

    ax : matplotlib.axes._axes.Axes
        A axis to put the text into

    facecolor: str
        The color of the label box background.

    """

    _plot_text(
        text=text,
        ax=ax,
        pos=(0.05, 0.95),
        text_options=dict(fontsize=7.3, color="black"),
        bbox=dict(facecolor=facecolor, alpha=0.8, edgecolor="black", boxstyle="round"),
    )


def example_matrix(
    dataset: GalaxyDataset,
    y_pred: ArrayLike,
    layout: tuple[int] = (3, 3),
    true_false_colors: tuple[str] = ("#40a02b", "#e64553"),
    true_false_secondary_colors: tuple[str] = ("#a6d189", "#e78284"),
    border_width: float = 3.0,
    figsize: tuple[float] = (7, 7),
    plot_args: dict = {},
    save_path: str | None = None,
    save_args: dict = {"bbox_inches": "tight"},
    seed: int | None = None,
):

    rng = np.random.default_rng(seed=seed)

    y_true = dataset.get_labels().cpu().numpy()
    n_examples = layout[0] * layout[1]

    mask = y_true == y_pred
    accuracy = np.sum(mask) / y_true.size

    n_true = int(np.round(n_examples * accuracy))
    n_false = int(np.round(n_examples * (1 - accuracy)))

    true_idx = rng.choice(np.argwhere(mask == True).ravel(), n_true)  # noqa: E712
    false_idx = rng.choice(np.argwhere(mask == False).ravel(), n_false)  # noqa: E712

    fig, ax = plt.subplots(*layout, figsize=figsize, layout="constrained")
    ax = ax.ravel()

    for i in range(0, n_examples):

        # Plot true predictions first
        if i < n_true:
            img, label = dataset[int(true_idx[i])]
            color_idx = 0
        else:
            img, label = dataset[int(false_idx[i - n_true])]
            color_idx = 1

        img = img.cpu().swapaxes(0, 2)
        label = label.cpu()
        pred = y_pred[int(false_idx[i - n_true])]

        ax[i] = _set_axes_border(
            ax=ax[i], color=true_false_colors[color_idx], width=border_width
        )
        ax[i].imshow(img, **plot_args)
        _plot_label(
            text=f"True: {CLASS_NAMES[int(label)]}\nPred: {CLASS_NAMES[int(pred)]}",
            ax=ax[i],
            facecolor=true_false_secondary_colors[color_idx],
        )

    if save_path is not None:
        fig.savefig(save_path, **save_args)
