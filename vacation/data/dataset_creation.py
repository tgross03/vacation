import gc
import shutil
import warnings
from hashlib import file_digest
from pathlib import Path

import h5py
import numpy as np
import requests
from sklearn.model_selection import train_test_split as split
from tqdm.auto import tqdm


def generate_dataset(location: Path, redownload: bool = False, overwrite: bool = True):
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


def extend_dataset(
    path: str,
    target_path: str,
    images: np.ndarray,
    labels: np.ndarray,
    copy_original: bool = True,
    overwrite: bool = False,
) -> None:
    target_path = Path(target_path)

    if not target_path.is_file():
        shutil.copy(Path(path), target_path)
    elif copy_original and not overwrite:
        raise FileExistsError(
            "This file already exists. Set 'overwrite = True' to overwrite the file!"
        )

    with h5py.File(target_path, mode="a") as hf:
        images_h5 = hf["images"]
        labels_h5 = hf["ans"]

        original_size = images_h5.shape[0]

        images_h5.resize(original_size + images.shape[0], axis=0)
        labels_h5.resize(original_size + labels.shape[0], axis=0)

        images_h5[original_size:] = images
        labels_h5[original_size:] = labels


def _save_h5(path: Path, X: np.ndarray, y: np.ndarray):
    if path.exists():
        extend_dataset(
            path=None,
            target_path=path,
            images=X,
            labels=y,
            copy_original=False,
            overwrite=True,
        )
    else:
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


def train_test_split(
    path: str,
    test_size: float = 0.2,
    test_type: str = "test",
    name_prefix: str | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: bool = True,
    pack_size: int = None,
    overwrite: bool = False,
):
    path = Path(path)

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

        if pack_size:
            training_pack_idx = zip(
                np.arange(0, train_idx.size, pack_size),
                np.arange(pack_size, train_idx.size + pack_size, pack_size),
            )
            test_pack_idx = zip(
                np.arange(0, train_idx.size, pack_size),
                np.arange(pack_size, train_idx.size + pack_size, pack_size),
            )
            train_idx = [train_idx[i:j] for i, j in training_pack_idx]
            test_idx = [test_idx[i:j] for i, j in test_pack_idx]
        else:
            train_idx = [train_idx]
            test_idx = [test_idx]

        train_path = path.parent / (path.stem + "_train.h5")

        test_stem = name_prefix if name_prefix else path.stem
        test_path = path.parent / (test_stem + f"_{test_type}.h5")

        if not train_path.exists() or overwrite:
            for idx_pgk in tqdm(train_idx, desc="Loading and saving training data"):
                X_train, y_train = (
                    np.asarray(hf["images"][idx_pgk], dtype=np.float32),
                    labels[idx_pgk],
                )

                _save_h5(path=train_path, X=X_train, y=y_train)

                del X_train
                del y_train
        else:
            print(
                "Training dataset already exists. Skipping creation. "
                "Set 'overwrite = True' to overwrite!"
            )

        if not test_path.exists() or overwrite:
            for idx_pgk in tqdm(test_idx, desc="Saving test data"):
                X_test, y_test = (
                    np.asarray(hf["images"][idx_pgk], dtype=np.float32),
                    labels[idx_pgk],
                )

                _save_h5(path=test_path, X=X_test, y=y_test)

                del X_test
                del y_test
        else:
            print(
                "Test dataset already exists. Skipping creation. "
                "Set'overwrite = True' to overwrite!"
            )

    return train_path, test_path
