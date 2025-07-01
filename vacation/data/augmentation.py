import h5py
import numpy as np
import torchvision.transforms.v2 as transforms
from scipy.signal import fftconvolve
import torch
from tqdm.auto import tqdm

from pathlib import Path

import shutil

from vacation.data import download_dataset

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def _numpy_to_tensor(array: np.ndarray, device: str = "cuda") -> torch.Tensor:
    return torch.from_numpy(array).permute(2, 0, 1).float().to(device)

def _tensor_to_numpy(tensor: torch.Tensor, device: str = "cuda") -> np.ndarray:
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()


_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
        transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0.5, hue=0),
    ]
)


def augment_dataset(path: str, class_index: int, target_count: int = 2600, device: str = "cuda", seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed=seed)

    with h5py.File(path, "r") as hf:

        labels = np.array(hf["ans"])
        images = hf["images"]

        original_indices = np.where(labels == class_index)[0]
        needed_count = target_count - len(original_indices)

        print(
            f"[Class {class_index}] Current: {len(original_indices)}, Adding: {needed_count}"
        )

        augmented_images = np.zeros((needed_count, *images[0].shape))
        augmented_labels = np.ones(needed_count, dtype=np.uint8) * class_index

        for i in tqdm(np.arange(needed_count), desc=f"Augmenting class {class_index}"):
            idx = rng.choice(original_indices)
            augmented_images[i] = _tensor_to_numpy(
                _transform(_numpy_to_tensor(images[idx], device=device)), device=device
            )

    return augmented_images, augmented_labels


def extend_dataset(
    path: str,
    target_path: str,
    images: np.ndarray,
    labels: np.ndarray,
    overwrite: bool = True,
) -> None:

    target_path = Path(target_path)

    if not target_path.is_file():
        shutil.copy(Path(path), target_path)
    elif not overwrite:
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