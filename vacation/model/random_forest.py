import numpy as np
import torchvision.transforms as T
from skimage.feature import hog
from tqdm.auto import tqdm


# extract histogram of oriented gradient features from image data and one sample image
def hog_features(
    df, length=None, pixels_per_cell=(12, 12), visualize=False, augmented=True
):

    if length is None:
        n = len(df)
    else:
        n = length

    # example image to see the effect of HOG and the filter

    sample_image = df[0][0]

    if augmented:
        gaussian_filter = T.GaussianBlur(3, sigma=1.0)
        sample_image = gaussian_filter(sample_image).permute(1, 2, 0).cpu().numpy()
    else:
        sample_image = sample_image.permute(1, 2, 0).cpu().numpy()

    sample_fd, sample_image = hog(
        sample_image,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1,
    )

    # initialization of feature vector
    X = np.zeros((n, sample_fd.shape[0]), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for i in tqdm(range(n)):
        image_tensor, label = df[i]
        if augmented:
            image_np = gaussian_filter(image_tensor).permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # compute HOG features without visualization
        fd = hog(
            image_np,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            visualize=visualize,
            channel_axis=-1,
        )
        X[i] = fd
        y[i] = label
    return X, y, sample_image
