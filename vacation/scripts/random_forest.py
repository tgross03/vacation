import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from vacation.data import GalaxyDataset
from vacation.model.random_forest import hog_features

parser = argparse.ArgumentParser(
    description="Extracting HOG features and hyperparameter optimization for a Random Forest."
)
# if --augmented is set in command line, agumented is set to True, otherwise False
parser.add_argument(
    "--augmented", action="store_true", help="Use augmented dataset if set."
)
args = parser.parse_args()

augmented = args.augmented
# initialize augmented datasets

train_ds_aug = GalaxyDataset(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_train.h5",
    device="cuda:1",
    max_cache_size="12G",
    cache_loaded=True,
)

valid_ds_aug = GalaxyDataset(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_valid.h5",
    device="cuda:1",
    max_cache_size="5G",
    cache_loaded=True,
)

# initialize original dataset
train_ds = GalaxyDataset(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_proc_train.h5",
    device="cuda:1",
    max_cache_size="16G",
    cache_loaded=True,
)

# augmented datasets already split in train and validation

location = Path("/scratch/tgross/vacation_models/random_forest/")
num_npy_files = len([file for file in location.glob("*.npy") if file.is_file()])

if num_npy_files < 4:
    print("Extracting HOG features from augmented datasets...")
    X_train_aug, y_train_aug, sample_image = hog_features(train_ds_aug)
    X_val_aug, y_val_aug, _ = hog_features(valid_ds_aug)

    # original dataset
    print("Extracting HOG features from original dataset...")
    X_train, y_train, _ = hog_features(train_ds, augmented=False)

    # saving features and labels
    np.save(str(location / "rf_features_train_aug.npy"), X_train_aug)
    np.save(str(location / "rf_labels_train_aug.npy"), y_train_aug)
    np.save(str(location / "rf_features_valid_aug.npy"), X_val_aug)
    np.save(str(location / "rf_labels_valid_aug.npy"), y_val_aug)
    np.save(str(location / "rf_features_train.npy"), X_train)
    np.save(str(location / "rf_labels_train.npy"), y_train)

else:
    # loading if not features already saved
    X_train_aug = np.load(str(location / "rf_features_train_aug.npy"))
    y_train_aug = np.load(str(location / "rf_labels_train_aug.npy"))
    X_val_aug = np.load(str(location / "rf_features_valid_aug.npy"))
    y_val_aug = np.load(str(location / "rf_labels_valid_aug.npy"))
    X_train = np.load(str(location / "rf_features_train.npy"))
    y_train = np.load(str(location / "rf_labels_train.npy"))


# hyperparameter tuning for Random Forest
print("Starting hyperparameter optimization for Random Forest...")
RF = RandomForestClassifier(random_state=42)

# parameter distribution for optimization
param_dist = {
    "n_estimators": randint(50, 400),
    "max_features": ["sqrt", "log2", None],
    "max_depth": randint(1, 20),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 2),
}

rand_search = RandomizedSearchCV(
    RF,
    param_distributions=param_dist,
    n_iter=75,
    cv=5,
    n_jobs=-1,
    verbose=10,
    random_state=42,
)
if augmented:
    rand_search.fit(X_train_aug, y_train_aug)
else:
    rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print("Best hyperparameters are:", rand_search.best_params_)

filename = str(location / "rf_optimized.sav")
joblib.dump(best_rf, filename)
