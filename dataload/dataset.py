from torch.utils.data import Dataset
import torch
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataframe_set, label_pair, transform=None, mode='train'):
        self.images_path = dataframe_set['img_path'].values
        self.masks_path = dataframe_set['mask_path'].values
        self.transform = transform
        self.label_pair = label_pair
        self.mode = mode

    def __len__(self):
        return len(self.masks_path)

    def __getitem__(self, idx):
        input_path = self.images_path[idx]
        label_path = self.masks_path[idx]

        input_value, labels = self._data_generation(input_path, label_path)

        return input_value, labels

    def _data_generation(self, input_path, label_path):
        label_pairs = list(range(1, len(self.label_pair) + 1))

        # Segmentation Gen

        input_value = cv2.imread(input_path)
        labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        new_masks = []

        for idx, label in enumerate(label_pairs):
            tmp = np.where(labels == idx, 1, 0)
            new_masks.append(tmp)

        new_masks = np.array(new_masks)
        new_masks = np.moveaxis(np.array(new_masks), 0, -1)

        if self.transform is not None:
            input_value = self.transform(image=input_value, mask=new_masks)
            input_value['mask'] = torch.moveaxis(input_value['mask'], 2, 0)

        return input_value['image'], input_value['mask']


