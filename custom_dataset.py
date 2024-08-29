from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, source_path, target_path, palette=None, transform=None):
        super(CustomDataset, self).__init__()
        self.transform = transform
        self.palette = palette
        self.source = sorted([os.path.join(source_path, file) for file in os.listdir(source_path)])
        self.target = sorted([os.path.join(target_path, file) for file in os.listdir(target_path)])
        filenames1 = {os.path.splitext(os.path.basename(path))[0] for path in self.source}
        filenames2 = {os.path.splitext(os.path.basename(path))[0] for path in self.target}
        self.source = [path for path in self.source if os.path.splitext(os.path.basename(path))[0] in filenames2]

    def __len__(self):
        return len(self.source)

    def tgt_image_preprocess(self, image):
        image = image.permute(1, 2, 0).numpy()

        target_mask_class = np.zeros((image.shape[0], image.shape[1]), dtype=int)

        image = (image*255).astype(int)

        for rgb, class_idx in self.palette.items():
            mask = np.all(image == rgb, axis=-1)
            target_mask_class[mask] = class_idx

        return target_mask_class

    def __getitem__(self, idx):
        src_image = Image.open(self.source[idx])
        tgt_image = Image.open(self.target[idx]).convert("RGB")
        if self.transform:
            src_image = self.transform(src_image)
            tgt_image = self.transform(tgt_image)
        tgt_image = self.tgt_image_preprocess(tgt_image)
        return src_image, tgt_image
