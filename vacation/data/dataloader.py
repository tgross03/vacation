import h5py
import torch

import numpy as np
import sys

import warnings

from torch.utils.data import DataLoader, Dataset

from pathlib import Path

# Data Caching adapted from own previous project: simtools (https://github.com/tgross03/simtools/blob/main/simtools/data/dataset.py)
# Originally licensed under MIT License. Copyright (c) 2024, Tom Groß.

VALID_UNIT_PREFIXES = {"K": 3, "M": 6, "G": 9, "T": 12, "P": 15}
CACHE_CLEANING_POLICIES = ["oldest", "youngest", "largest"]

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

                self._max_size = (
                    int(max_size[:-1]) * 10 ** (VALID_UNIT_PREFIXES[max_size[-1]])
                )
            case int():
                self._max_size = max_size
            case _:
                raise TypeError(
                    "Only str or float are valid types for the maximum size!"
                )

        if cleaning_policy not in CACHE_CLEANING_POLICIES:
            raise ValueError(
                f"The given cleaning policy does not exist! Valid values are {CACHE_CLEANING_POLICIES}"
            )

        self.cleaning_policy = cleaning_policy

    def __getitem__(self, i):
        record = (
            self._records[str(i)] if str(i) in self._records else None
        )

        if record is None:
            return None

        return record["data"][0]

    def __len__(self):
        return len(self._records)

    def add(self, uid, data):
        record = dict(
            data=data,
            size=int(
                data[0].nelement() * data[0].element_size()
                + sys.getsizeof(data[0])
                + np.sum([sys.getsizeof(d) for d in data[1]])
                + sys.getsizeof(data[1])
                + sys.getsizeof(data[2])
            ),
        )

        MAX_ITER = 10
        while self._memsize + record["size"] > self._max_size:
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
        match self.cleaning_policy:
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

        return self(self.max_size, self.cleaning_policy)


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
    "Edge-on with Bulge"
]

class GalaxyDataset(Dataset):
    def __init__(self, path: Path | str, device: str,         
                 cache_loaded: bool = True,
                 max_cache_size: str = "3G",
                 cache_cleaning_policy: str = "oldest",
                ):
        super(GalaxyDataset, self).__init__()
        self.path: Path = path if isinstance(path, Path) else Path(path)
        self.device: str = device
        
        self._hf: h5py._hl.files.File = h5py.File(self.path, "r")
        self._labels = torch.from_numpy(self._hf["ans"][:]).to(self.device)

        if cache_loaded:
            self._cache = DataCache(max_cache_size, cache_cleaning_policy)
        else:
            self._cache = None

    def __len__(self) -> int:
        return self._hf["images"].shape[0]

    def __getitem__(self, key: int) -> tuple[torch.Tensor, int]:

        if not isinstance(key, int):
            raise KeyError("The given key has to be an integer value!")

        if self._cache is not None:
            cached_data = self._cache[key]
            if cached_data is not None:
                return cached_data, self._labels[key]

        data = torch.Tensor(self._hf["images"][key].T).to(self.device)
        
        if self._cache is not None:
            self._cache.add(key, data)
        
        return data, self._labels[key]

    def get_label(self, key: int) -> tuple(int, str):

        if not isinstance(key, int):
            raise KeyError("The given key has to be an integer value!")

        label = int(self._labels[key])

        return label, CLASS_NAMES[label]
        
        