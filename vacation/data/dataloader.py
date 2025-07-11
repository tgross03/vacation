import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
    def __init__(self, max_size, cleaning_policy):
        """
        Creates a new data cache for files in the dataset.

        The cache records are structured as follows:
        records = { UNIQUE_ID: { data: DATA, size: MEMORY_SIZE }, ... }

        The data has to have the following form:
        DATA = (torch.Tensor)

        Parameters
        ----------

        max_size: str or int
            The maximum memory size of the file cache. Can be given as bytes (int)
            or a string (e.g. `"10G"` or `"1M"`).

        cleaning_policy: str
            The way the program will remove files from the cache if it has reached its
            memory limit.
            You can choose from:
                1. `oldest` -> the oldest file in the cache will be removed
                2. `youngest` -> the youngest file in the cache will be removed
                3. `largest` -> the largest file in the cache will be removed
        """
        self._records = dict()
        self._memsize = 0
        self._timeline = np.array([], dtype=int)

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

        return record["data"]

    def __len__(self):
        return len(self._records)

    def add(self, uid, data):
        record = dict(
            data=data,
            size=int(sys.getsizeof(data.untyped_storage())),
        )

        try:
            free_memory = torch.cuda.mem_get_info(device=data.device)[0]
        except ValueError:
            free_memory = np.infty

        MAX_ITER = 10
        while (
            self._memsize + record["size"] > self._max_size
            or free_memory < record["size"] * 1.5
        ):
            if MAX_ITER <= 0:
                warnings.warn(
                    f"The record with the uid {uid} could not be cached because it is too large! "
                    f"Current cache size: {self._memsize}"
                )

            self.clean()
            MAX_ITER -= 1

        self._timeline = np.append(self._timeline, uid)
        self._records[str(uid)] = record
        self._memsize += record["size"]

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
        self._records.pop(str(uid), None)

    def clear(self):
        del self._records
        del self._timeline
        del self._memsize

        return self(self._max_size, self._cleaning_policy)


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
    ):
        super(GalaxyDataset, self).__init__()
        self.path: Path = path if isinstance(path, Path) else Path(path)
        self.device: str = device

        self._hf: h5py._hl.files.File = h5py.File(self.path, "r")
        self._labels: torch.Tensor = torch.from_numpy(self._hf["ans"][:]).to(self.device)
        self._end_idx: int = end_index
        self._index_collection: np.typing.ArrayLike | None = index_collection

        if cache_loaded:
            self._cache = DataCache(max_cache_size, cache_cleaning_policy)
        else:
            self._cache = None

    def __len__(self) -> int:
        if self._index_collection is not None:
            return len(self._index_collection)
        else:
            return self._hf["images"].shape[0] if self._end_idx == -1 else self._end_idx + 1

    def __getitem__(self, key: int) -> tuple[torch.Tensor, int]:

        if not isinstance(key, int):
            raise KeyError("The given key has to be an integer value!")

        if key >= len(self):
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

    def get_labels(self) -> torch.Tensor:
        return self._labels
