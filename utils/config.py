import torch
from torchvision import transforms

class Config:
    TRAIN_PATH = "/home/etman/etman/python/projects/brain_tumor_classification/data/Training"
    VAL_PATH = "/home/etman/etman/python/projects/brain_tumor_classification/data/Testing"

    # Hyperparameters
    EPOCHS = 50
    LR = 0.0001
    BATCH_SIZE = 32


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 512
    NUM_CLASSES = 4
    CLASSES_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    TRANSFORMS_DICT = {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),              # standard input size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],             # ImageNet stats (for pretrained models)
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }