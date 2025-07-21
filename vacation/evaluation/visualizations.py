import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.metrics import ConfusionMatrixDisplay

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

    _plot_text(
        text=text,
        ax=ax,
        pos=(0.05, 0.95),
        text_options=dict(fontsize=7.3, color="black"),
        bbox=dict(facecolor=facecolor, alpha=0.8, edgecolor="black", boxstyle="round"),
    )


def plot_example_matrix(
    dataset: GalaxyDataset,
    y_pred: torch.Tensor,
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

    y_pred = y_pred.cpu().numpy()

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
            pred = y_pred[int(true_idx[i])]
            color_idx = 0
        else:
            img, label = dataset[int(false_idx[i - n_true])]
            pred = y_pred[int(false_idx[i - n_true])]
            color_idx = 1

        img = img.cpu().swapaxes(0, 2)
        label = label.cpu()

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

    return fig, ax


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    cmap: str = "inferno",
    normalize: bool = False,
):
    return ConfusionMatrixDisplay.from_predictions(
        y_true=y_true.cpu().numpy(),
        y_pred=y_pred.cpu().numpy(),
        cmap="inferno",
        normalize="true" if normalize else None,
        values_format=".2f" if normalize else None,
    )


def plot_hyperparameter_importance(study: optuna.study.Study, log: bool = True):

    param_importances = optuna.importance.get_param_importances(study=study)

    fig, ax = plt.subplots(figsize=(3, 6))
    ax.barh(
        y=list(param_importances.keys())[::-1],
        width=list(param_importances.values())[::-1],
        color="#e64553",
        log=True,
    )
    ax.set_xlabel("fANOVA Importance Score")
    ax.set_ylabel("Hyperparameter")

    return fig, ax
