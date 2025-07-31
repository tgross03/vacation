# VACATION
**V**isu**a**l Galaxy **C**l**a**ssification Using Convolu**tio**nal **N**eural Networks

## Introduction

In this project we used two methods to classify images from the *Galaxy10 DECaLS* dataset.
One method is a Convolutional Neural Network. The other one is a Random Forest.

<img width="2282" height="636" alt="cnn_structure" src="https://github.com/user-attachments/assets/35f6128d-9127-4f75-96e7-ce904376a86c" />

Image created by [@nicktky](https://github.com/nicktky)

## Dataset

This project uses the *Galaxy10 DECaLS dataset* by *Leung, W. Henry* and *Bovy, Jo*,
which can be found under the following DOI: https://doi.org/10.5281/zenodo.10845025.

## Installation

This project is created as an installable python package for python versions `>= 3.12`.

To install it, you first have to clone this repository.
After that you can execute the following command inside of the cloned directory:


```
$ pip install -e .
```

> [!IMPORTANT]
> The required packages to execute the commands and scripts are installed automatically.
> Some of the scripts require the availability of a pytorch supported GPU.

## Usage / Reproduction of Results

> [!IMPORTANT]  
> To use the results of our training and optimization, you can use the files in the `best_models` directory in this repository.
> It contains
> 1. The `vacation.sqlite3` file, which contains the Optuna optimization studies.
> 2. The `vacation_v2.py` file, which is the exported best CNN model of the Optuna study.
> 3. The `rf_optimized.zip` archive, which contains the `joblib` dump of the best Random Forest model

The training and analysis of the project can be reproduced using the built in Command-Line Interface (CLI)
of this project.
With that you are able to 
- Create and process the datasets
- Start the hyperparameter optimization of the CNN
- Start the hyperparameter optimization of the Random Forest
- Create visualizations for the dataset, hyperparameter optimization and the evaluation

If you are unsure about the usage of a command or its arguments, you can use the `--help` flag
in order to get an overview of the command.

Example:

```shell
$ vacation --help
$ vacation optim cnn --help
```

## Creating and Processing the Dataset

To create and process the dataset you will first have to choose a directory where
the data should be created.

> [!WARNING]
> The entire dataset collection requires a disk space of about **8.6 GiB**.

Enter your chosen directory and execute the following command:

```shell
$ vacation dataset create ./
```

Feel free to adjust the arguments of this command like the memory consumption.

> [!TIP]
> If you want the dataset to overwrite or redownload parts of the dataset, use the `--overwrite` or `--redownload` flags.

## Starting the CNN Optimization

> [!IMPORTANT]
> These steps require GPU support!

After downloading and generating all necessary datasets, you can proceed to start the Optuna hyperparameter optimization.
For that go to a directory where you want your Optuna study to be saved. This won't take up much disk space.

Then you can use

```shell
$ vacation optim cnn PATH/TO/TRAIN_DATASET PATH/TO/VALID_DATASET --checkpoint-dir PATH/TO/DESIRED/CHECKPOINT_DIR
```

For the train and validation dataset, provide the paths of the created files `Galaxy10_DECals_train.h5` and `Galaxy10_DECals_valid.h5` datasets
you created previously.
The `--checkpoint-dir` path can be chosen freely. This can also take up some disk space but not as much as the datasets, only about 1.2 GiB.

## Starting the RF Optimization

> [!IMPORTANT]
> These steps require GPU support!

To start the Random Forest optimization, use the following command at in an arbitrary location:

```shell
$ vacation optim rf PATH/TO/TRAIN_DATASET PATH/TO/VALID_DATASET PATH/TO/NON_AUGMENTED_TRAIN_DATASET --checkpoint-dir PATH/TO/DESIRED/CHECKPOINT_DIR
```

For the train and validation dataset, provide the paths of the created files `Galaxy10_DECals_train.h5` and `Galaxy10_DECals_valid.h5` datasets
you created previously. The non augmented training dataset can be found in the same directory with the name `Galaxy10_DECals_proc_train`.

The `--checkpoint-dir` path can be chosen freely. This can also take up some disk space but not as much as the datasets, only about 515 MiB.


## Using the Visualizations

The CLI has multiple visualizations:
- Plot of the class distribution, example images and augmentation examples of a HDF5 dataset using `vacation dataset plot`
- Plot of a HOG feature extraction example with `vacation dataset hog`
- Plots of the Random Forest test evaluation results with `vacation rf eval`
- Plots of the CNN hyperparameter optimization results with `vacation cnn plot_metric` and `vacation optim plot_importance`
- Plots of the CNN test evaluation results with `vacation cnn eval`

For further information on these commands use the `--help` flag.

> [!TIP]
> The test dataset can be found under the file name `Galaxy10_DECals_proc_test.h5` in the data directory.

> [!WARNING]
> Exported CNN models (`.pt` files) contain a parameter determining the location of the train and validation datasets on the system the model was created on.
> If you want to use the provided models, you will have to provide these values yourself. There should be some kind of `--dataset-directory` option that you
> have to set to your dataset directory (not file!). If you are using the functions from the code itself, you can provide the datasets as arguments to the
> functions!
