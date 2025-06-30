import gc
import shutil
import warnings
from hashlib import file_digest
from pathlib import Path

import h5py
import numpy as np
import requests
from sklearn.model_selection import train_test_split as split


def create_dataset(location: Path, redownload: bool = False, overwrite: bool = True):

    url = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"

    if not isinstance(location, Path):
        location = Path(location)

    location.mkdir(exist_ok=True, parents=True)
    target = location / url.split("/")[-1]

    if target.is_file() and not redownload:
        warnings.warn(
            "The data file already exists at this location!"
            "Use a different location or set 'redownload=True'."
        )
    else:
        target.unlink(missing_ok=True)
        target.touch()
        print(f"Downloading data from {url}. Saving to {str(target)}.")

        with requests.get(url, stream=True) as req:
            with open(target, "wb") as file:
                shutil.copyfileobj(req.raw, file)

        print("Finished download of data!")

    with open(target, "rb") as file:
        md5sum = file_digest(file, "md5").hexdigest()

    if md5sum != "c6b7b4db82b3a5d63d6a7e3e5249b51c":
        warnings.warn(
            "The MD5 checksum of the downloaded file is not equal to the original file! "
            "Consider retrying the download!"
        )
    else:
        print("Checksum passed successfully.")

    target_proc = Path(target.parent) / (target.stem + "_proc.h5")

    if target_proc.is_file():
        if overwrite:
            target_proc.unlink()
        elif not overwrite:
            raise FileExistsError(
                "This file already exists. Set 'overwrite = True' to overwrite the file!"
            )

    gc.collect()

    with h5py.File(target, "r") as hf:

        print("Reading image data ...")
        images = np.asarray(hf["images"], dtype=np.float32)
        print(
            f"Completed reading image data (Memory size {images.nbytes * 1e-6} MB) ..."
        )

        images /= 255.0

        print("Reading label data ...")
        labels = np.asarray(hf["ans"], dtype=np.uint8)
        print(
            f"Completed reading label data (Memory size {labels.nbytes * 1e-6} MB) ..."
        )

    with h5py.File(target_proc, "w") as hf:
        print("Writing image data ...")
        hf.create_dataset(
            "images",
            data=images,
            maxshape=(None, *images.shape[1:]),
            chunks=True,
            compression="gzip",
        )

        del images

        print("Writing label data ...")
        hf.create_dataset(
            "ans",
            data=labels,
            maxshape=(None, *labels.shape[1:]),
            chunks=True,
            compression="gzip",
        )

        del labels

    gc.collect()


def train_test_split(
    path: str,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: bool = True,
):

    path = Path(path)

    def _save_h5(path: Path, suffix: str, X: np.ndarray, y: np.ndarray):
        path = path.parent / (path.stem + f"_{suffix}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "images",
                data=X,
                maxshape=(None, *X.shape[1:]),
                chunks=True,
                compression="gzip",
            )
            f.create_dataset(
                "ans",
                data=y,
                maxshape=(None, *y.shape[1:]),
                chunks=True,
                compression="gzip",
            )
        return path

    with h5py.File(path, "r") as hf:

        labels = np.asarray(hf["ans"], dtype=np.uint8)

        train_idx, test_idx = split(
            np.arange(labels.shape[0]),
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=labels if stratify else None,
        )

        print(
            f"Splitted index arrays: train -> {train_idx.size} | test -> {test_idx.size}"
        )

        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        print("Loading test data ...")
        X_train, y_train = (
            np.asarray(hf["images"][train_idx], dtype=np.float32),
            labels[train_idx],
        )
        print("Loaded training data ... Saving to file ...")
        train_hf = _save_h5(path=path, suffix="train", X=X_train, y=y_train)

        del X_train
        del y_train

        print("Loading training data ...")
        X_test, y_test = (
            np.asarray(hf["images"][test_idx], dtype=np.float32),
            labels[test_idx],
        )
        print("Loaded test data ... Saving to file ...")
        test_hf = _save_h5(path=path, suffix="train", X=X_test, y=y_test)

        del X_test
        del y_test

    return train_hf, test_hf
