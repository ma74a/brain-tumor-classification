import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import optim
from torch.nn import CrossEntropyLoss

from utils.config import Config
from src.load_data import load_data
from src.model import BrainTumorModel
from src.training import train_and_val
from utils.visualization import plot_losses, plot_confusion_matrix


def main():
    train_loader, val_loader = load_data()

    model = BrainTumorModel(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = CrossEntropyLoss()

    model, train_losses, train_accuracies, val_losses, val_accuracies, cm = train_and_val(model,
                                                                                          train_loader,
                                                                                          val_loader,
                                                                                          optimizer,
                                                                                          criterion,
                                                                                          Config.DEVICE,
                                                                                          Config.EPOCHS)
    
    plot_losses(train_losses, val_losses)
    plot_confusion_matrix(cm, Config.CLASSES_NAMES)


if __name__ == "__main__":
    main()