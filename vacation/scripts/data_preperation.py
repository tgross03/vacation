from vacation.data import augment_dataset, generate_dataset, train_test_split

# Download dataset and generate pre-processed ()
try:
    generate_dataset(
        "/scratch/tgross/vacation_data/reduced_size/", overwrite=False, redownload=False
    )
except FileExistsError:
    pass

# Train-Test split
train_test_split(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_proc.h5",
    random_state=42,
    pack_size=1000,
    overwrite=False,
)

# Augmenting training data
augment_dataset(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_proc_train.h5",
    target_path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_augmented_train.h5",
    seed=42,
    overwrite=False,
)

# Train-Validation split
train_test_split(
    path="/scratch/tgross/vacation_data/reduced_size/Galaxy10_DECals_augmented_train.h5",
    name_prefix="Galaxy10_DECals",
    test_type="valid",
    random_state=1337,
    pack_size=4000,
    overwrite=True,
)
