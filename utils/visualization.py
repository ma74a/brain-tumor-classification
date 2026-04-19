import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_losses(train_losses, val_losses):
    """
    Plot training vs validation loss.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, class_names=None, normalize=False):
    """
    Plot confusion matrix.

    Args:
        cm (np.array): confusion matrix
        class_names (list): list of class names
        normalize (bool): normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.show()


# Optional helper to save plots

def save_plot_losses(train_losses, val_losses, path="loss_plot.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def save_confusion_matrix(cm, class_names=None, path="cm.png", normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.savefig(path)
    plt.close()
