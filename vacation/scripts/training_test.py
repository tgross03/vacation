from vacation.model import VCNN
from vacation.data import GalaxyDataset
import numpy as np
import torch

rng = np.random.default_rng(1337)

train_ds = GalaxyDataset(
    path="/scratch/tgross/vacation_data/Galaxy10_DECals_train.h5",
    device="cuda:1",
    max_cache_size="15G",
    cache_loaded=True,
    # index_collection=rng.integers(0, 16813, 10000),
)

valid_ds = GalaxyDataset(
    path="/scratch/tgross/vacation_data/Galaxy10_DECals_valid.h5",
    device="cuda:1",
    max_cache_size="4G",
    cache_loaded=True,
    # index_collection=rng.integers(0, 4204, 3000),
)

model = VCNN(
    train_batch_size=128,
    valid_batch_size=128,
    out_channels=[4, 6, 8],
    dropout_rates=[0.4, 0.4, 0.4],
    lin_out_features=[300],
    optimizer=torch.optim.AdamW,
    activation_func=torch.nn.PReLU,
    learning_rate=0.001,
    weight_decay=0.03,
    loss_func=torch.nn.CrossEntropyLoss,
    device="cuda:1"
)
model.init_data(train_dataset=train_ds, valid_dataset=valid_ds)

model.train_epochs(n_epochs=60)

model.save(".models/model_full_v2_2.pt", relative_to_package=True)
