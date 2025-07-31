import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Data Caching adapted from own previous project: simtools
# (https://github.com/tgross03/simtools/blob/main/simtools/data/dataset.py)
# Originally licensed under MIT License. Copyright (c) 2024, Tom Groß.

VALID_UNIT_PREFIXES = {"K": 3, "M": 6, "G": 9, "T": 12, "P": 15}
CACHE_CLEANING_POLICIES = ["oldest", "youngest", "largest"]

_unit_prefix = ["", "k", "M", "G", "T", "P", "E"]


def format_bytes(byte: int) -> str:
    closest_base = np.floor(np.log10(byte))
    prefix = _unit_prefix[np.max([int(closest_base // 3), 0])]
    return f"{byte * 10**(-(closest_base - closest_base % 3))} {prefix}B"


class DataCache:
    def __init__(
        self,
        max_size: str | int,
        device: str,
        cleaning_policy: str,
        use_cpu_memory: bool,
    ):
        """
        Creates a new data cache for files in the dataset.

        The cache records are structured as follows:
        records = { UNIQUE_ID: { data: DATA, size: MEMORY_SIZE, device: 'cuda' | 'cpu' }, ... }

        The data has to have the following form:
        DATA = (torch.Tensor)

        Parameters
        ----------

        max_size: str or int
            The maximum memory size of the file cache. Can be given as bytes (int)
            or a string (e.g. `"10G"` or `"1M"`).

        device: str
            The primary device to send data to.

        cleaning_policy: str
            The way the program will remove files from the cache if it has reached its
            memory limit.
            You can choose from:
                1. `oldest` -> the oldest file in the cache will be removed
                2. `youngest` -> the youngest file in the cache will be removed
                3. `largest` -> the largest file in the cache will be removed

        use_cpu_memory: bool
            Whether to move cleaned items to the normal memory (RAM) if the vRAM is
            full instead of removing the object from the memory.
        """
        self._records = dict()
        self._memsize = 0
        self._timeline = np.array([], dtype=int)
        self._use_cpu_memory = use_cpu_memory
        self._device = device

        match max_size:
            case str():
                if max_size[-1] not in VALID_UNIT_PREFIXES:
                    raise ValueError(
                        f"The valid unit prefixes are: {VALID_UNIT_PREFIXES.keys()}"
                    )

                self._max_size = int(max_size[:-1]) * 10 ** (
                    VALID_UNIT_PREFIXES[max_size[-1]]
                )
            case int():
                self._max_size = max_size
            case _:
                raise TypeError(
                    "Only str or float are valid types for the maximum size!"
                )

        if cleaning_policy not in CACHE_CLEANING_POLICIES:
            raise ValueError(
                f"The given cleaning policy does not exist! "
                f"Valid values are {CACHE_CLEANING_POLICIES}"
            )

        self._cleaning_policy = cleaning_policy

    def __getitem__(self, i):
        record = self._records[str(i)] if str(i) in self._records else None

        if record is None:
            return None

        if record["device"] != self._device:
            if self._cleaning_loop(uid=i, record=record):
                record["data"].to(self._device)
                record["device"] = self._device
                self._add(uid=i, record=record, metadata_only=True)
            else:
                return None

        return record["data"]

    def __len__(self):
        return len(self._records)

    def _is_memory_available(
        self,
        record: dict,
        memory_multiplier: float = 0.9,
        record_multiplier: float = 1.5,
    ):
        try:
            free_memory = torch.cuda.mem_get_info(device=self._device)[0]
        except Exception:
            free_memory = np.inf

        return (
            self._memsize + record["size"] < self._max_size
            and free_memory * memory_multiplier > record["size"] * record_multiplier
        )

    def _cleaning_loop(self, uid: int, record: dict, max_iter: int = 10, **check_args):
        while not self._is_memory_available(record=record, **check_args):
            if max_iter <= 0:
                warnings.warn(
                    f"The record with the uid {uid} could not be cached because it is too large! "
                    f"Current cache size: {self._memsize}"
                )
                break

            self.clean()
            max_iter -= 1

        return max_iter > 0

    def _add(self, uid: int, record: dict, metadata_only: bool = False):
        self._timeline = np.append(self._timeline, uid)
        self._memsize += record["size"]
        if not metadata_only:
            self._records[str(uid)] = record

    def add(self, uid: int, data: torch.Tensor):

        record = dict(
            data=data,
            size=int(sys.getsizeof(data.untyped_storage())),
            device=str(data.device),
        )

        if self._cleaning_loop(uid=uid, record=record):
            if str(record["data"].device) != self._device:
                data.to(self._device)
                record["device"] = self._device
            self._add(uid=uid, record=record)
        else:
            record["data"].to("cpu")
            record["device"] = "cpu"

    def clean(self):
        uid = 0
        match self._cleaning_policy:
            case "oldest":
                uid = self._timeline[0]
            case "youngest":
                uid = self._timeline[-1]
            case "largest":
                largest = [0, 0]
                for id, record in self._records.items():
                    if record["size"] > largest[1]:
                        largest[0] = int(id)
                        largest[1] = record["size"]
                uid = largest[0]

        record = self._records[str(uid)]

        self._timeline = np.delete(self._timeline, np.where(self._timeline == uid))
        self._memsize -= record["size"]

        if self._use_cpu_memory:
            record["data"].to("cpu")
            record["device"] = "cpu"
        else:
            self._records.pop(str(uid), None)
            del record

    def clear(self):
        del self._records
        del self._timeline
        del self._memsize

        return self.__init__(
            max_size=self._max_size,
            cleaning_policy=self._cleaning_policy,
            device=self._device,
            use_cpu_memory=self._use_cpu_memory,
        )


CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]


class GalaxyDataset(Dataset):
    def __init__(
        self,
        path: Path | str,
        device: str,
        cache_loaded: bool = True,
        index_collection: np.typing.ArrayLike | None = None,
        end_index: int = -1,
        max_cache_size: str = "3G",
        cache_cleaning_policy: str = "oldest",
        use_cpu_memory: str = True,
    ):
        super(GalaxyDataset, self).__init__()
        self.path: Path = path if isinstance(path, Path) else Path(path)
        self.device: str = device

        self._hf: h5py._hl.files.File = h5py.File(self.path, "r")
        self._labels: torch.Tensor = torch.from_numpy(self._hf["ans"][:]).to(
            self.device
        )
        self._end_idx: int = end_index
        self._index_collection: np.typing.ArrayLike | None = index_collection

        if cache_loaded:
            self._cache = DataCache(
                max_size=max_cache_size,
                device=device,
                cleaning_policy=cache_cleaning_policy,
                use_cpu_memory=use_cpu_memory,
            )
        else:
            self._cache = None

    def __len__(self) -> int:
        if self._index_collection is not None:
            return len(self._index_collection)
        else:
            return (
                self._hf["images"].shape[0]
                if self._end_idx == -1
                else self._end_idx + 1
            )

    def __getitem__(self, key: int) -> tuple[torch.Tensor, int]:

        if not (isinstance(key, int) or isinstance(key, slice)):
            raise KeyError("The given key has to be an integer value!")

        if not isinstance(key, slice) and key >= len(self):
            raise KeyError("The given key is too large!")

        if self._index_collection is not None:
            key = self._index_collection[key]

        if self._cache is not None:
            cached_data = self._cache[key]
            if cached_data is not None:
                return cached_data, self._labels[key]

        data = torch.Tensor(self._hf["images"][key].T).to(self.device)

        if self._cache is not None:
            self._cache.add(key, data)

        return data, self._labels[key]

    def get_label(self, key: int) -> tuple[int, str]:
        if not isinstance(key, int):
            raise KeyError("The given key has to be an integer value!")

        label = int(self._labels[key])

        return label, CLASS_NAMES[label]

    def get_images(self, device: str) -> torch.Tensor:
        if self._index_collection is None:
            if self._end_idx == -1:
                return self._labels
            else:
                return self._labels[: self._end_idx]
        else:
            return self._labels[self._index_collection]

    def get_labels(self) -> torch.Tensor:
        if self._index_collection is None:
            if self._end_idx == -1:
                return self._labels
            else:
                return self._labels[: self._end_idx]
        else:
            return self._labels[self._index_collection]

    def get_args(self) -> dict:
        return {
            "cache_loaded": self._cache is not None,
            "index_collection": self._index_collection,
            "end_index": self._end_idx,
            "max_cache_size": self._cache._max_size if self._cache else "3G",
            "cache_cleaning_policy": self._cache._cleaning_policy
            if self._cache
            else "oldest",
        }

    def plot_examples(
        self,
        show_percentages: bool = False,
        random_state: int | None = None,
        save_path: str | None = None,
        save_args: dict = {"bbox_inches": "tight"},
    ):

        rng = np.random.default_rng(seed=random_state)

        labels, counts = np.unique(self.get_labels().cpu(), return_counts=True)
        total = np.sum(counts)

        label_idx = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for label in labels:
            label_idx[label] = np.argwhere(self.get_labels().cpu() == label)[0][
                rng.integers(0, counts[label])
            ]

        fig, ax = plt.subplots(2, 5, figsize=(10, 5), layout="tight")
        ax = ax.ravel()

        for i in tqdm(range(len(labels)), desc="Plotting images"):

            label, count = labels[i], counts[i]

            image = self[int(label_idx[label])][0].cpu().permute(1, 2, 0)
            if np.isclose(image.max(), 255.0):
                image /= 255.0
            ax[label].imshow(image)
            ax[label].set_title(f"{CLASS_NAMES[label]} ({label})", fontsize="small")

            if show_percentages:
                from vacation.evaluation.visualizations import _plot_label

                _plot_label(
                    f"{('{:.1f}').format(np.round(100 * count / total, 1))}%",
                    ax=ax[label],
                    facecolor="white",
                    fontsize=10,
                )

            ax[label].axis("off")

        if save_path:
            fig.savefig(save_path, **save_args)

    def plot_distribution(
        self,
        plot_args: dict = {
            "rwidth": 0.96,
            "facecolor": "#e64553",
        },
        save_path: str | None = None,
        save_args: dict = {"bbox_inches": "tight"},
    ):

        fig, ax = plt.subplots(layout="tight")

        labels, counts = np.unique(self.get_labels().cpu(), return_counts=True)
        _, bins, bars = ax.hist(
            self.get_labels().cpu(),
            bins=np.arange(0, 11),
            label=f"Total: {len(self)}",
            **plot_args,
        )
        ax.bar_label(bars)
        ax.set_xticks(bins[:-1] + 0.5)
        ax.set_xticklabels([plt.Text(bins[label] + 0.5, 0, label) for label in labels])
        ax.set_xlabel("Classes")
        ax.set_ylabel("Counts")
        ax.set_ylim(0, np.max(counts) * 1.1)
        ax.legend()

        if save_path:
            fig.savefig(save_path, **save_args)

        return fig, ax
