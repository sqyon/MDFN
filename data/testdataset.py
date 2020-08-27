import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """Dataset for test."""

    def load_config(self, param):
        self.data_path = param['data_path']
        self.view_size = param['view_size']
        self.sr_rate = param['sr_rate']
        self.store_in_memory = param['store_in_memory']

    def __init__(self, param):
        super().__init__()
        self.load_config(param)
        self.image_name_list = os.listdir(self.data_path)
        self.ground_truth = []
        self.len = len(self.image_name_list)

        # Load data:
        if self.store_in_memory:
            for image_name in self.image_name_list:
                image_path = f'{self.data_path}/{image_name}'
                load_data = np.load(image_path)
                self.ground_truth.append(load_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.store_in_memory:
            ground_truth = self.ground_truth[index]
        else:
            image_path = f'{self.data_path}/{self.image_name_list[index]}'
            ground_truth = np.load(image_path)
        ground_truth = ground_truth[:, :, :, :, 0]  # Only select Y channel

        # Crop:
        self.height, self.width = ground_truth.shape[2], ground_truth.shape[3]
        self.height -= self.height % self.sr_rate
        self.width -= self.width % self.sr_rate
        view_height_st = (ground_truth.shape[0] - self.view_size) // 2
        view_width_st = (ground_truth.shape[1] - self.view_size) // 2
        ground_truth = ground_truth[view_height_st: view_height_st + self.view_size,
                                    view_width_st: view_width_st + self.view_size, :self.height, :self.width]

        # Blur
        blured = np.zeros(
            [self.view_size, self.view_size, self.height // self.sr_rate, self.width // self.sr_rate],
            dtype=ground_truth.dtype)
        for i in range(self.view_size):
            for j in range(self.view_size):
                blur = cv2.blur(ground_truth[i, j, :, :], (self.sr_rate, self.sr_rate))
                blured[i, j, :, :] = blur[self.sr_rate // 2::self.sr_rate, self.sr_rate // 2::self.sr_rate]
        ground_truth, blured = ground_truth[np.newaxis, :], blured[np.newaxis, :]
        ground_truth = torch.from_numpy(ground_truth.astype(np.float32))
        blured = torch.from_numpy(blured.astype(np.float32))
        ret = {
            'gt': ground_truth,
            'input': blured,
            'name': self.image_name_list[index],
        }
        return ret
