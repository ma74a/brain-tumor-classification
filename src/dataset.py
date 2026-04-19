import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset

import os
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import Config

class TumorDataset(Dataset):
    def __init__(self, data_dir: str, transforms=None) -> None:
        self.data_dir = data_dir
        self.transforms = transforms

        self.images_paths = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}

        self._load_images()

    def _load_images(self):
        for label, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_dir = os.path.join(self.data_dir, class_name)
            self.class_to_idx[class_name] = label
            self.classes.append(class_name)
            for img_name in os.listdir(class_dir):
                if not img_name.endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(class_dir, img_name)
                self.images_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.images_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("L")
        if self.transforms:
            img = self.transforms(img)

        return img, label
                
                                        

# if __name__ == "__main__":

#     ob = TumorDataset(Config.TRAIN_PATH, Config.TRANSFORMS_DICT)
#     print(ob.classes)
#     print(ob.class_to_idx)
#     print(len(ob))
#     img, label = ob[0]
#     print(img.size())
#     print(label)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()