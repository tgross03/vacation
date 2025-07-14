import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score  # , f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary
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


class VCNN(nn.Module):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        out_channels: list[int],
        dropout_rates: list[float],
        lin_out_features: list[int],
        optimizer: optim.Optimizer,
        activation_func: Callable,
        learning_rate: float,
        weight_decay: float,
        img_size: int = 256,
        num_labels: int = 10,
        loss_func: Callable = torch.nn.CrossEntropyLoss,
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

        self._out_channels: list[int] = out_channels
        self._dropout_rates: list[float] = dropout_rates
        self._lin_out_features: list[int] = lin_out_features
        self._activation_func: Callable = activation_func
        self._lr: float = learning_rate
        self._weight_decay: float = weight_decay

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

        # Model definition
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self._out_channels[0],
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=self._out_channels[0]),
            self._activation_func(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            nn.Dropout2d(p=self._dropout_rates[0]),
            nn.Conv2d(
                in_channels=self._out_channels[0],
                out_channels=self._out_channels[1],
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=self._out_channels[1]),
            self._activation_func(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            nn.Dropout2d(p=self._dropout_rates[1]),
            nn.Flatten(),
            nn.LazyLinear(out_features=self._lin_out_features[0]),
            self._activation_func(),
            nn.Dropout(p=self._dropout_rates[2]),
            nn.Linear(
                in_features=self._lin_out_features[0],
                out_features=self._num_labels,
            ),
        )

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

        self.model.train()
        prog = tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            colour="#8caaee",
            disable=not show_progress,
        )
        for X, y in prog:

            self._optimizer.zero_grad()

            y_pred = self.model(X)
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
        self.model.eval()
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
                y_pred = self.model(X)

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
        trial: optuna.trial.Trial | None,
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

            if trial.should_prune:
                raise optuna.exceptions.TrialPruned()

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
            "out_channels": self._out_channels,
            "dropout_rates": self._dropout_rates,
            "lin_out_features": self._lin_out_features,
            "optimizer": str(self._optimizer.__class__.__name__),
            "activation_func": str(self._activation_func.__name__),
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "loss_func": str(self._loss_func.__class__.__name__),
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

    def summarize(self, input_dims: tuple[int] = (3, 256, 256)):
        self.model.eval()
        return summary(self, input_dims)

    @classmethod
    def load(
        cls,
        path: str,
        train_dataset: GalaxyDataset | None = None,
        valid_dataset: GalaxyDataset | None = None,
        metrics: typing.Dict[str, [Callable, dict]] = DEFAULT_METRICS,
        relative_to_package: bool = False,
    ) -> "VCNN":

        if relative_to_package:
            path = str(Path(vacation.__file__).parent.parent / path)

        state = torch.load(path, weights_only=False)

        cls = cls(
            train_batch_size=state["train_batch_size"],
            valid_batch_size=state["valid_batch_size"],
            out_channels=state["out_channels"],
            dropout_rates=state["dropout_rates"],
            lin_out_features=state["lin_out_features"],
            optimizer=getattr(optim, state["optimizer"]),
            activation_func=getattr(torch.nn, state["activation_func"]),
            learning_rate=state["lr"],
            weight_decay=state["weight_decay"],
            img_size=state["img_size"],
            num_labels=state["num_labels"],
            loss_func=getattr(torch.nn, state["loss_func"]),
            metrics=metrics,
            seed=state["seed"],
            device=state["device"],
        )

        for key, metric in cls._metrics.items():
            cls._metrics[key].train_vals = state[key][0]
            cls._metrics[key].valid_vals = state[key][1]

        cls._loss_metric.train_vals = state["loss"][0]
        cls._loss_metric.valid_vals = state["loss"][1]

        model_state_dict = state["model_state_dict"]
        model_state_dict = {
            f"model.{key}": value for key, value in model_state_dict.items()
        }

        cls.load_state_dict(model_state_dict)

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
