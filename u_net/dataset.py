from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self, train_image_path, train_mask_path, transform = None):
        super(dataset, self).__init__()
        self.transform = transform
        self.train_image_paths = sorted(glob(train_image_path))
        self.train_mask_paths = sorted(glob(train_mask_path))

    def __len__(self):
        return len(self.train_image_paths)

    def __getitem__(self, item):
        image_path = self.train_image_paths[item]
        mask_path = self.train_mask_paths[item]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = np.transpose(image, (2, 0, 1))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
        mask[mask==255.0] = 1.0
        mask = np.expand_dims(mask, axis = 0)

        if self.transform:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmentations = self.transform(image=image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


