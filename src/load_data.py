import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from .dataset import TumorDataset
from utils.config import Config


def load_data():
    train_dataset = TumorDataset(Config.TRAIN_PATH, Config.TRANSFORMS_DICT["train"])
    val_dataset = TumorDataset(Config.VAL_PATH, Config.TRANSFORMS_DICT["val"])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader