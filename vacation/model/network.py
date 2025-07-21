import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torchinfo
from sklearn.metrics import accuracy_score  # , f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import vacation
from vacation.data import GalaxyDataset


@dataclass
class Metric:
    _func: Callable | None
    func_args: dict = ({},)
    train_vals: torch.Tensor = torch.Tensor([])
    valid_vals: torch.Tensor = torch.Tensor([])

    def func(self, y_true, y_pred) -> float:
        return self._func(y_true=y_true, y_pred=y_pred, **self.func_args)

    def append(
        self, train_val: float | None = None, valid_val: float | None = None
    ) -> None:
        if train_val:
            self.train_vals = torch.cat(
                [self.train_vals, torch.Tensor([train_val]).flatten()]
            )
        if valid_val:
            self.valid_vals = torch.cat(
                [self.valid_vals, torch.Tensor([valid_val]).flatten()]
            )

    def as_exportable(self) -> list[np.ndarray]:
        return [self.train_vals, self.valid_vals]


DEFAULT_METRICS = {
    "accuracy": [accuracy_score, {}],
    # "precision": [precision_score, {"average": "samples"}],
    # "recall": [recall_score, {"average": "samples"}],
    # "f1": [f1_score, {"average": "samples"}],
}


def _calculate_conv_size(dim, kernel_size, padding, stride):
    return ((dim - kernel_size + 2 * padding) / stride) + 1


@dataclass
class ConvBlock:
    in_channels: int
    out_channels: int
    conv_kernel_size: int
    conv_padding: int
    conv_stride: int
    activation_func: callable
    pool_kernel_size: int
    pool_padding: int
    pool_stride: int
    dropout_rate: float

    def get_layers(self):
        return [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
                stride=self.conv_stride,
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
            self.activation_func(),
            nn.MaxPool2d(
                kernel_size=self.pool_kernel_size,
                padding=self.pool_padding,
                stride=self.pool_stride,
            ),
            nn.Dropout2d(p=self.dropout_rate),
        ]

    def get_post_block_dim(self, dim: int) -> list[int]:
        sizes = []

        if dim < self.conv_kernel_size:
            raise ValueError(
                f"The image size ({dim}x{dim}) is smaller than the convolution kernel "
                f"({self.conv_kernel_size}x{self.conv_kernel_size})!"
            )

        sizes.append(
            _calculate_conv_size(
                dim=dim,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
                stride=self.conv_stride,
            )
        )

        if sizes[-1] < self.pool_kernel_size:
            raise ValueError(
                f"The image size ({sizes[-1]}x{sizes[-1]}) is smaller than the pooling "
                f"kernel ({self.pool_kernel_size}x{self.pool_kernel_size})!"
            )

        sizes.append(
            _calculate_conv_size(
                dim=sizes[-1],
                kernel_size=self.pool_kernel_size,
                padding=self.pool_padding,
                stride=self.pool_stride,
            )
        )

        if not sizes[-1].is_integer() or not sizes[-2].is_integer():
            raise ValueError(
                f"An image sizes {sizes[-2]} or {sizes[-1]} are not integer values!"
            )

        return sizes


class VCNN(nn.Module):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        num_conv_blocks: int,
        out_channels: list[int],
        conv_dropout_rates: list[float],
        num_dense_layers: int,
        lin_out_features: list[int],
        lin_dropout_rates: list[float],
        optimizer: optim.Optimizer,
        activation_func: Callable,
        learning_rate: float,
        weight_decay: float,
        img_size: int = 128,
        num_labels: int = 10,
        loss_func: Callable = torch.nn.CrossEntropyLoss,
        conv_kernel_args={"kernel_size": 3, "padding": 0, "stride": 1},
        pool_kernel_args={"kernel_size": 2, "padding": 1, "stride": 2},
        verbose: bool = False,
        metrics: typing.Dict[str, [Callable, dict]] = DEFAULT_METRICS,
        seed: int | None = None,
        device: str = "cuda",
    ):
        super().__init__()

        # Dataset attributes
        self._img_size: int = img_size
        self._num_labels: int = num_labels

        # Optimizable Hyperparameters
        self._train_batch_size: int = train_batch_size
        self._valid_batch_size: int = valid_batch_size

        self._num_conv_blocks: int = num_conv_blocks
        self._out_channels: list[int] = out_channels.copy()
        self._conv_dropout_rates: list[float] = conv_dropout_rates.copy()

        self._num_dense_layers: int = num_dense_layers
        self._lin_out_features: list[int] = lin_out_features.copy()
        self._lin_dropout_rates: list[float] = lin_dropout_rates.copy()

        self._activation_func: Callable = activation_func
        self._lr: float = learning_rate
        self._weight_decay: float = weight_decay

        self._conv_kernel_args: dict = conv_kernel_args
        self._pool_kernel_args: dict = pool_kernel_args

        # Training parameters
        self._loss_func: Callable = loss_func()

        rng = np.random.default_rng(seed=seed)
        self._seed: int = rng.integers(low=0, high=2**32 - 1)
        self._device: str = device

        # Metrics
        metric_keys = list(metrics.keys())
        metrics = [
            Metric(_func=func[0], func_args=func[1]) for key, func in metrics.items()
        ]

        self._metrics: typing.Dict[str, Metric] = dict(zip(metric_keys, metrics))

        self._loss_metric: Metric = Metric(_func=None)

        layers = []
        out_channels.insert(0, 3)

        # Add convolution blocks
        post_block_dims = []
        for i in range(0, self._num_conv_blocks):
            block = ConvBlock(
                in_channels=out_channels[i],
                out_channels=out_channels[i + 1],
                conv_kernel_size=conv_kernel_args["kernel_size"],
                conv_padding=conv_kernel_args["padding"],
                conv_stride=conv_kernel_args["stride"],
                activation_func=self._activation_func,
                pool_kernel_size=pool_kernel_args["kernel_size"],
                pool_padding=pool_kernel_args["padding"],
                pool_stride=pool_kernel_args["stride"],
                dropout_rate=self._conv_dropout_rates[i],
            )
            layers.extend(block.get_layers())
            post_block_dims.extend(
                block.get_post_block_dim(
                    dim=post_block_dims[-1] if i > 0 else self._img_size
                )
            )
            if verbose:
                print(
                    f"-------------- CONV BLOCK {i+1} -------------- \n"
                    f"POST-CONV DIM: {post_block_dims[-2]} | POST-POOL DIM: {post_block_dims[-1]}"
                )

        # Add fully connected layers
        dense_layers = []

        if self._num_dense_layers == 0:
            raise ValueError("The minimum number of dense layers is 1!")

        for i in range(0, self._num_dense_layers):
            # Use LazyLinear if first layer
            if i == 0:
                dense_layers.extend(
                    [
                        nn.LazyLinear(out_features=self._lin_out_features[0]),
                        self._activation_func(),
                        nn.Dropout(p=self._lin_dropout_rates[0]),
                    ]
                )
            # Use normal Linear layer otherwise
            else:
                dense_layers.extend(
                    [
                        nn.Linear(
                            in_features=self._lin_out_features[i - 1],
                            out_features=self._lin_out_features[i],
                        ),
                        self._activation_func(),
                        nn.Dropout(p=self._lin_dropout_rates[i]),
                    ]
                )

        layers.extend(
            [
                nn.Flatten(),
                *dense_layers,
                nn.Linear(
                    in_features=self._lin_out_features[-1],
                    out_features=self._num_labels,
                ),
            ]
        )

        # Model definition
        self.model = nn.Sequential(*layers)

        # Init parameters and set model device
        torch.manual_seed(self._seed)
        self.to(self._device)

        self._optimizer: optim.Optimizer = optimizer(
            self.parameters(), lr=self._lr, weight_decay=weight_decay
        )

        self._epoch = 0
        self._train_dataset: GalaxyDataset | None = None
        self._valid_dataset: GalaxyDataset | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def init_data(
        self,
        train_dataset: GalaxyDataset | str,
        valid_dataset: GalaxyDataset | str,
        train_args: dict = {},
        valid_args: dict = {},
    ) -> None:

        if isinstance(train_dataset, str):
            self._train_dataset = GalaxyDataset(
                path=train_dataset, device=self._device, **train_args
            )
        else:
            self._train_dataset = train_dataset

        if isinstance(valid_dataset, str):
            self._valid_dataset = GalaxyDataset(
                path=valid_dataset, device=self._device, **valid_args
            )
        else:
            self._valid_dataset = valid_dataset

    def _train_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        show_progress: bool = True,
    ) -> None:

        metric_vals = dict(
            zip(self._metrics.keys(), [torch.Tensor()] * len(self._metrics.keys()))
        )

        loss_vals = torch.Tensor()

        self.train()
        prog = tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            colour="#8caaee",
            disable=not show_progress,
        )
        for X, y in prog:

            self._optimizer.zero_grad()

            y_pred = self(X)
            loss = self._loss_func(y_pred, y)
            loss.backward()

            self._optimizer.step()

            # Calculate metrics
            for name, metric in self._metrics.items():
                metric_vals[name] = torch.cat(
                    [
                        metric_vals[name],
                        torch.Tensor(
                            [
                                metric.func(
                                    y_true=y.detach().cpu(),
                                    y_pred=y_pred.detach().cpu().argmax(dim=1),
                                )
                            ]
                        ),
                    ]
                )

            loss_vals = torch.cat([loss_vals, loss.cpu().detach()[None]])

            prog.set_postfix(
                self._get_progress_postfix(
                    loss=loss_vals.mean(), accuracy=metric_vals["accuracy"].mean()
                )
            )

        # Append average of metric values from all training steps
        for name, metric in self._metrics.items():
            metric.append(train_val=metric_vals[name].mean())

        # Append loss value
        self._loss_metric.append(train_val=loss_vals.mean())

    def _valid_epoch(
        self,
        epoch: int,
        valid_loader: torch.utils.data.DataLoader,
        show_progress: bool = True,
    ) -> None:
        self.eval()
        metric_vals = dict(
            zip(self._metrics.keys(), [torch.Tensor()] * len(self._metrics.keys()))
        )

        loss_vals = torch.Tensor()

        with torch.no_grad():
            prog = tqdm(
                valid_loader,
                desc=f"Validating epoch {epoch}",
                colour="#ca9ee6",
                disable=not show_progress,
            )
            for X, y in prog:
                y_pred = self(X)

                loss_vals = torch.cat(
                    [
                        loss_vals,
                        self._loss_func(y_pred.detach().cpu(), y.detach().cpu())[None],
                    ]
                )

                # Calculate metrics
                for name, metric in self._metrics.items():
                    metric_vals[name] = torch.cat(
                        [
                            metric_vals[name],
                            torch.Tensor(
                                [
                                    metric.func(
                                        y_true=y.detach().cpu(),
                                        y_pred=y_pred.detach().cpu().argmax(dim=1),
                                    )
                                ]
                            ),
                        ]
                    )

                prog.set_postfix(
                    self._get_progress_postfix(
                        loss=loss_vals.mean(), accuracy=metric_vals["accuracy"].mean()
                    )
                )

            # Append average of metric values from all validation steps
            for name, metric in self._metrics.items():
                metric.append(valid_val=metric_vals[name].mean())

            self._loss_metric.append(valid_val=loss_vals.mean())

    def _get_progress_postfix(self, loss: torch.Tensor, accuracy: torch.Tensor) -> dict:
        return {
            "Loss": "{:.3f}".format(loss.numpy()),
            "Accuracy": "{:.3f}".format(accuracy.numpy()),
        }

    def train_epochs(
        self,
        n_epochs: int,
        stop_early: bool = True,
        patience: int = 5,
        min_delta: float = 0.0,
        cumultative_delta: bool = False,
        trial: optuna.trial.Trial | None = None,
        trial_metric: str = "accuracy",
        save_path: str | None = None,
        save_interval: int = 1,
        show_progress: bool = True,
    ) -> None:

        if not self._train_dataset or not self._valid_dataset:
            raise AttributeError(
                "There were no datasets initialized! "
                "Use the VCNN.init_data method to initialize the data!"
            )

        train_loader = DataLoader(
            self._train_dataset, batch_size=self._train_batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            self._valid_dataset, batch_size=self._valid_batch_size, shuffle=True
        )

        epochs_no_increase = 0

        for i in torch.arange(0, n_epochs):
            self._epoch += 1
            self._train_epoch(
                epoch=self._epoch,
                train_loader=train_loader,
                show_progress=show_progress,
            )
            self._valid_epoch(
                epoch=self._epoch,
                valid_loader=valid_loader,
                show_progress=show_progress,
            )

            if i % save_interval == 0 and save_path is not None:
                self.save(path=save_path)

            if trial is not None:
                trial.report(
                    self._metrics[trial_metric].valid_vals[-1].cpu().numpy(),
                    self._epoch,
                )

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if stop_early and self._epoch >= 2:
                comp_idx = -(epochs_no_increase + 2) if cumultative_delta else -2
                if (
                    self._loss_metric.valid_vals[-1]
                    <= self._loss_metric.valid_vals[comp_idx] - min_delta
                ):
                    epochs_no_increase = 0
                else:
                    epochs_no_increase += 1

                if epochs_no_increase > patience:
                    break

    def save(self, path: str, relative_to_package: bool = False) -> None:

        if not self._train_dataset or not self._valid_dataset:
            raise AttributeError(
                "There were no datasets initialized! "
                "Use the VCNN.init_data method to initialize the data!"
            )

        state = {
            "epoch": self._epoch,
            # Dataset attributes
            "img_size": self._img_size,
            "num_labels": self._num_labels,
            "train_dataset": str(self._train_dataset.path.absolute()),
            "train_args": self._train_dataset.get_args(),
            "valid_dataset": str(self._valid_dataset.path.absolute()),
            "valid_args": self._valid_dataset.get_args(),
            # Optimizable Hyperparameters
            "train_batch_size": self._train_batch_size,
            "valid_batch_size": self._valid_batch_size,
            "num_conv_blocks": self._num_conv_blocks,
            "out_channels": self._out_channels,
            "conv_dropout_rates": self._conv_dropout_rates,
            "num_dense_layers": self._num_dense_layers,
            "lin_out_features": self._lin_out_features,
            "lin_dropout_rates": self._lin_dropout_rates,
            "optimizer": str(self._optimizer.__class__.__name__),
            "activation_func": str(self._activation_func.__name__),
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "loss_func": str(self._loss_func.__class__.__name__),
            "conv_kernel_args": self._conv_kernel_args,
            "pool_kernel_args": self._pool_kernel_args,
            # Model & Optimizer
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            # Loss values
            "loss": self._loss_metric.as_exportable(),
            # Utils
            "model_structure": str(self.model),
            "seed": self._seed,
            "device": self._device,
        }

        for key, metric in self._metrics.items():
            state[key] = metric.as_exportable()

        if relative_to_package:
            path = Path(vacation.__file__).parent.parent / path
            path.parent.mkdir(exist_ok=True, parents=True)
            path = str(path)

        torch.save(state, path)

    def summarize(
        self, input_dims: tuple[int] | None = None
    ) -> torchinfo.model_statistics.ModelStatistics:
        self.eval()
        input_dims = (
            input_dims
            if input_dims is not None
            else (self._train_batch_size, 3, self._img_size, self._img_size)
        )
        return torchinfo.summary(self, input_dims)

    def predict_dataset(
        self,
        dataset: GalaxyDataset,
        show_progress: bool = True,
        return_true: bool = False,
    ) -> torch.Tensor:
        self.eval()
        y_pred = torch.Tensor([]).to(self._device)

        for i in tqdm(
            range(0, len(dataset)), desc="Predicting dataset", disable=not show_progress
        ):

            y_pred = torch.cat([y_pred, self(dataset[i][0][None]).argmax(dim=1)])

        if return_true:
            return y_pred, dataset.get_labels()
        else:
            return y_pred

    def plot_metric(
        self,
        key: str,
        components: list[str] = ["train", "valid"],
        colors: list[str] = ["#1e66f5", "#e64553"],
        name: str | None = None,
        plot_args: dict = {},
        save_args: dict = {"bbox_inches": "tight"},
    ):

        fig, ax = plt.subplots(1, 1, layout="constrained")

        metric = self._loss_metric if key == "loss" else self._metrics[key]

        if "train" in components:
            ax.plot(metric.train_vals, label="Train", color=colors[0])
        if "valid" in components:
            ax.plot(metric.valid_vals, label="Valid", color=colors[1])

        if name is None:
            name = key.title()

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)

        ax.legend()

        return fig, ax

    @classmethod
    def load(
        cls,
        path: str,
        train_dataset: GalaxyDataset | None = None,
        valid_dataset: GalaxyDataset | None = None,
        load_model_state: bool = True,
        load_optimizer_state: bool = True,
        metrics: typing.Dict[str, [Callable, dict]] = DEFAULT_METRICS,
        relative_to_package: bool = False,
        device: str | None = None,
    ) -> "VCNN":

        if relative_to_package:
            path = str(Path(vacation.__file__).parent.parent / path)

        state = torch.load(path, weights_only=False)

        cls = cls(
            train_batch_size=state["train_batch_size"],
            valid_batch_size=state["valid_batch_size"],
            num_conv_blocks=state["num_conv_blocks"],
            out_channels=state["out_channels"],
            conv_dropout_rates=state["conv_dropout_rates"],
            num_dense_layers=state["num_dense_layers"],
            lin_out_features=state["lin_out_features"],
            lin_dropout_rates=state["lin_dropout_rates"],
            optimizer=getattr(optim, state["optimizer"]),
            activation_func=getattr(torch.nn, state["activation_func"]),
            learning_rate=state["lr"],
            weight_decay=state["weight_decay"],
            img_size=state["img_size"],
            num_labels=state["num_labels"],
            loss_func=getattr(torch.nn, state["loss_func"]),
            conv_kernel_args=state["conv_kernel_args"],
            pool_kernel_args=state["pool_kernel_args"],
            metrics=metrics,
            seed=state["seed"],
            device=state["device"] if device is None else device,
        )

        for key, metric in cls._metrics.items():
            cls._metrics[key].train_vals = state[key][0]
            cls._metrics[key].valid_vals = state[key][1]

        cls._loss_metric.train_vals = state["loss"][0]
        cls._loss_metric.valid_vals = state["loss"][1]

        if load_model_state:
            cls.model.load_state_dict(state["model_state_dict"])

        if load_optimizer_state:
            cls._optimizer.load_state_dict(state["optimizer_state_dict"])

        if train_dataset is None and valid_dataset is None:
            cls.init_data(
                train_dataset=state["train_dataset"],
                valid_dataset=state["valid_dataset"],
                train_args=state["train_args"],
                valid_args=state["valid_args"],
            )
        else:
            cls.init_data(train_dataset=train_dataset, valid_dataset=valid_dataset)

        cls._epoch = state["epoch"]

        return cls
