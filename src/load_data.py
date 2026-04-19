from torch.utils.data import DataLoader

from .dataset import TumorDataset
from utils.config import Config


def load_data():
    train_dataset = TumorDataset(Config.TRAIN_PATH, Config.TRANSFORMS_DICT["train"])
    val_dataset = TumorDataset(Config.VAL_PATH, Config.TRANSFORMS_DICT["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader