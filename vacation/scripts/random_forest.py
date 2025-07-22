import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.filters import gaussian
from vacation.data import GalaxyDataset
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import argparse
import joblib 

parser = argparse.ArgumentParser(description='Extracting HOG features and hyperparameter optimization for a Random Forest.')
#if --augmented is set in command line, agumented is set to True, otherwise False
parser.add_argument('--augmented', action='store_true', help='Use augmented dataset if set.')
args = parser.parse_args()

augmented = args.augmented
#initialize augmented datasets

train_ds_aug = GalaxyDataset(
    path="../../.data/Galaxy10_DECals_train.h5",
    device="cpu",
    max_cache_size="7G",
    cache_loaded=False,
)

valid_ds_aug = GalaxyDataset(
    path="../../.data/Galaxy10_DECals_valid.h5",
    device="cpu",
    max_cache_size="2G",
    cache_loaded=False,
)

#initialize original dataset
train_ds = GalaxyDataset(
    path="../../.data/Galaxy10_DECals_proc_train.h5",
    device="cpu",
    max_cache_size="7G",
    cache_loaded=False,
)


#extract histogram of oriented gradient features from image data and one sample image
def hog_features(df, length=None, pixels_per_cell=(12, 12), visualize=False, augmented=True):

    if length == None:
        n = len(df)
    else:
        n = length

    #example image to see the effect of HOG and the filter

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

    #initialization of feature vector
    X = np.zeros((n, sample_fd.shape[0]), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for i in tqdm(range(n)):
        image_tensor, label = df[i]
        if augmented:
            image_np = gaussian_filter(image_tensor).permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    #compute HOG features without visualization
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

#augmented datasets already split in train and validation
print("Extracting HOG features from augmented datasets...")
X_train_aug, y_train_aug, sample_image = hog_features(train_ds_aug)
X_val_aug, y_val_aug, _ = hog_features(valid_ds_aug)

#original dataset
print("Extracting HOG features from original dataset...")
X_train, y_train = hog_features(train_ds, augmented=False)

#saving features and labels
np.save("rf_features_train_aug.npy",X_train_aug)
np.save("rf_labels_train_aug.npy",y_train_aug)
np.save("rf_features_valid_aug.npy", X_val_aug)
np.save("rf_labels_valid_aug.npy", y_val_aug)
np.save("rf_features_train.npy",X_train)
np.save("rf_labels_train.npy",y_train)

#loading if not features already saved

X_train_aug = np.load("../notebooks/rf_features_train_aug.npy")
y_train_aug = np.load("../notebooks/rf_labels_train_aug.npy")
X_val_aug = np.load("../notebooks/rf_features_valid_aug.npy")
y_val_aug = np.load("../notebooks/rf_labels_valid_aug.npy")
X_train = np.load("../notebooks/rf_features_train.npy")
y_train = np.load("../notebooks/rf_labels_train.npy")


#hyperparameter tuning for Random Forest
print("Starting hyperparameter optimization for Random Forest...")
RF = RandomForestClassifier(
    random_state=42)

#parameter distribution for optimization
param_dist = {"n_estimators": randint(50,400),
              "max_features": ["sqrt","log2", None],
              "max_depth": randint(1,20),
              "min_samples_split": randint(2,10),
              "min_samples_leaf": randint(1,2),
              }

rand_search = RandomizedSearchCV(RF,
                                 param_distributions = param_dist,
                                 n_iter = 1, 
                                 cv = 3,
                                 n_jobs = -1,
                                 verbose = 10,
                                 random_state=42
                                )
if augmented:
    rand_search.fit(X_train_aug, y_train_aug)
else:
    rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print("Best hyperparameters are:", rand_search.best_params_)

filename = 'rf_optimized.sav'
joblib.dump(best_rf, filename)


