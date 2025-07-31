from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from vacation.data import CLASS_NAMES, GalaxyDataset


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


def _plot_label(text: str, ax, facecolor: str, fontsize=7.3):

    _plot_text(
        text=text,
        ax=ax,
        pos=(0.05, 0.95),
        text_options=dict(fontsize=fontsize, color="black"),
        bbox=dict(facecolor=facecolor, alpha=0.8, edgecolor="black", boxstyle="round"),
    )


def plot_example_matrix(
    dataset: GalaxyDataset,
    y_pred: torch.Tensor,
    layout: tuple[int] = (3, 3),
    true_false_ratio: float = 0.4,
    true_false_colors: tuple[str] = ("#a6d189", "#e78284"),
    border_width: float = 3.0,
    figsize: tuple[float] = (7, 7),
    plot_args: dict = {},
    save_path: str | Path | None = None,
    save_args: dict = {"bbox_inches": "tight"},
    seed: int | None = None,
):

    rng = np.random.default_rng(seed=seed)

    y_pred = y_pred.cpu().numpy()

    y_true = dataset.get_labels().cpu().numpy()
    n_examples = layout[0] * layout[1]

    mask = y_true == y_pred

    n_true = int(np.round(n_examples * true_false_ratio))
    n_false = int(np.round(n_examples * (1 - true_false_ratio)))

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

        ax[i].axis("off")

        ax[i].imshow(img, **plot_args)
        _plot_label(
            text=f"True: {CLASS_NAMES[int(label)]}\nPred: {CLASS_NAMES[int(pred)]}",
            ax=ax[i],
            facecolor=true_false_colors[color_idx],
            fontsize=7.3,
        )

    if save_path is not None:
        fig.savefig(save_path, **save_args)

    return fig, ax


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    cmap: str = "viridis",
    normalize: bool = False,
    save_path: str | Path | None = None,
    save_args: dict = {"bbox_inches": "tight"},
):
    cmatrix = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true.cpu().numpy(),
        y_pred=y_pred.cpu().numpy(),
        cmap=cmap,
        normalize="true" if normalize else None,
        values_format=".2f" if normalize else None,
    )

    if save_path is not None:
        cmatrix.figure_.savefig(save_path, **save_args)

    return cmatrix


def plot_hyperparameter_importance(
    study: optuna.study.Study,
    evaluator: str = "ped-anova",
    log: bool = True,
    save_path: str | Path | None = None,
    save_args: dict = {"bbox_inches": "tight"},
):

    match evaluator:
        case "ped-anova":
            eval_cls = optuna.importance.PedAnovaImportanceEvaluator()
            eval_name = "PED-ANOVA"
        case "fanova":
            eval_cls = optuna.importance.FanovaImportanceEvaluator()
            eval_name = "fANOVA"
        case "mean_decrease_impurity":
            eval_cls = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
            eval_name = "Mean Decrease Impurity"

    param_importances = optuna.importance.get_param_importances(
        study=study,
        evaluator=eval_cls,
    )

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.barh(
        y=list(param_importances.keys())[::-1],
        width=list(param_importances.values())[::-1],
        color="#e64553",
        log=log,
    )
    ax.set_xlabel(f"{eval_name} Importance Score")
    ax.set_ylabel("Hyperparameter")

    if save_path is not None:
        fig.savefig(save_path, **save_args)

    return fig, ax
