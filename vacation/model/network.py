import typing
from collections.abc import Callable
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from tqdm.auto import tqdm


@dataclass
class Metric:
    func: Callable | None
    train_vals: torch.Tensor = torch.Tensor([])
    valid_vals: torch.Tensor = torch.Tensor([])

    def append(self, train_val: float | None, valid_val: float | None):
        if train_val:
            self.train_vals = torch.cat([self.train_vals, torch.Tensor([train_val])])
        if valid_val:
            self.valid_vals = torch.cat([self.valid_vals, torch.Tensor([valid_val])])


DEFAULT_METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
}


class VCNN(nn.Module):
    def __init__(
        self,
        img_size: int,
        num_labels: int,
        train_batch_size: int,
        valid_batch_size: int,
        out_channels: list[int],
        dropout_rates: list[float],
        lin_out_features: list[int],
        optimizer: optim.Optimizer,
        activation_func: Callable,
        learning_rate: float,
        loss_func: Callable = torch.nn.CrossEntropyLoss,
        metrics: typing.Dict[str, Callable] = DEFAULT_METRICS,
        seed: int = 42,
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
        self._optimizer: optim.Optimizer = optimizer
        self._activation_func: Callable = (activation_func,)
        self._lr: float = learning_rate

        # Training parameters
        self._loss_func: Callable = loss_func()
        self._seed: int = seed
        self._device: str = device

        # Metrics

        metric_keys = list(metrics.keys())
        metrics = [Metric(func=func) for key, func in metrics.items()]

        self._metrics: typing.Dict[str, Metric] = dict(zip(metric_keys, metrics))

        self._loss_metric: Metric = Metric(func=None)

        # Model definition
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self._out_channels[0],
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            self._activation_func(),
            nn.Dropout2d(p=self._dropout_rates[0]),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2),
            nn.Conv2d(
                in_channels=self._out_channels[0],
                out_channels=self._out_channels[1],
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            self._activation_func(),
            nn.Dropout2d(p=self._dropout_rates[1]),
            nn.MaxPool2d(kernel_size=3, padding=2, stride=2),
            nn.Conv2d(
                in_channels=self._out_channels[1],
                out_channels=self._out_channels[2],
                kernel_size=3,
                padding=0,
                stride=1,
            ),
            self._activation_func(),
            nn.Dropout2d(p=self._dropout_rates[2]),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=self._lin_out_features[0]),
            self._activation_func(),
            nn.Linear(
                in_features=self._lin_out_features[0],
                out_features=self._lin_out_features[1],
            ),
            self._activation_func(),
            nn.Linear(
                in_features=self.self._lin_out_features[1],
                out_features=self._num_labels,
            ),
        )

        # Init parameters and set model device
        torch.manual_seed(self._seed)
        self.to(self._device)

        def forward(self, X: torch.Tensor):
            return self.model(X)

        def _train_epoch(self, epoch: int, train_loader: torch.utils.data.DataLoader):

            metric_vals = dict(
                zip(self._metrics.keys(), [torch.Tensor()] * len(self._metrics.keys()))
            )

            for X, y in tqdm(train_loader, desc=f"Training epoch {epoch}"):
                self._optimizer.zero_grad()

                y_pred = self._model(X)
                loss = self._loss_func(y_pred, y)
                loss.backward()

                self._optimizer.step()

                # Calculate metrics
                for name, metric in self._metrics.items():
                    metric_vals[name] = torch.cat(
                        [metric_vals[name], metric.func(y_true=y, y_pred=y_pred)]
                    )

            # Append average of metric values from all training steps
            for name, metric in self._metrics.items():
                metric.append(train_vals=metric_vals[name].mean())

            # Append loss value
            self._loss_metric.append(train_val=loss.cpu().detach())

        def _valid_epoch(epoch: int, valid_loader: torch.utils.data.DataLoader):
            self._model.eval()
            metric_vals = dict(
                zip(self._metrics.keys(), [torch.Tensor()] * len(self._metrics.keys()))
            )

            loss_vals = torch.Tensor()

            with torch.no_grad():
                for X, y in tqdm(valid_loader, desc=f"Validating epoch {epoch}"):
                    y_pred = self._model(X)

                    loss_vals = torch.cat([loss_vals, self._loss_func(y_pred, y)])

                    # Calculate metrics
                    for name, metric in self._metrics.items():
                        metric_vals[name] = torch.cat(
                            [metric_vals[name], metric.func(y_true=y, y_pred=y_pred)]
                        )

                # Append average of metric values from all validation steps
                for name, metric in self._metrics.items():
                    metric.append(valid_val=metric_vals[name].mean())

                self._loss_metric.append(valid_val=loss_vals.mean())

        def train_epochs(
            self,
            n_epochs: int,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            save_name: str | None = None,
            checkpoint_path: str | None = None,
            summarize: bool = True,
        ):
            for i in torch.arange(1, self.n_epochs + 1):
                self._train_epoch(epoch=i, train_loader=train_loader)
                self._valid_epoch(epoch=i, valid_loader=valid_loader)
