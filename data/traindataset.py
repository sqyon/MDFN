import cv2
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """Dataset for training."""

    def load_config(self, param):
        self.data_path = param['data_path']
        self.view_size = param['view_size']
        self.sr_rate = param['sr_rate']
        self.crop_size = param['crop_size'] * self.sr_rate
        self.store_in_memory = param['store_in_memory']
        self.repeat_rate = param['repeat_rate']
        assert type(self.repeat_rate) == int

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
                if len(load_data.shape) != 5:
                    # For 130LF dataset.
                    load_data = load_data[:, :, 15:-15, 15:-15]
                else:
                    load_data = load_data[:, :, :, :, 0]  # Only need Y channel.
                self.ground_truth.append(load_data)

    def __len__(self):
        return self.len * self.repeat_rate

    def augmentation(self, x):
        v_h, v_w, h, w = list(x.shape)
        x = np.transpose(x, [0, 2, 1, 3])
        x = x.reshape([v_h * h, v_w * w])
        x = np.rot90(x, random.choice(range(4)))
        if random.choice([0, 1]) == 0:
            x = np.flip(x, random.choice([0, 1]))
        x = x.reshape([v_h, h, v_w, w])
        x = np.transpose(x, [0, 2, 1, 3])
        return x

    def __getitem__(self, index):
        index %= self.len
        if not self.store_in_memory:
            image_path = f'{self.data_path}/{self.image_name_list[index]}'
            ground_truth = np.load(image_path)
            if len(ground_truth.shape) != 5:
                ground_truth = ground_truth[:, :, 15:-15, 15:-15]
            else:
                ground_truth = ground_truth[0]
        else:
            ground_truth = self.ground_truth[index]

        # Crop:
        view_height_st = random.choice(range(ground_truth.shape[0] - self.view_size + 1))
        view_width_st = random.choice(range(ground_truth.shape[1] - self.view_size + 1))
        pixel_height_st = random.choice(range(ground_truth.shape[2] - self.crop_size + 1))
        pixel_width_st = random.choice(range(ground_truth.shape[3] - self.crop_size + 1))
        ground_truth = ground_truth[view_height_st: view_height_st + self.view_size,
                                    view_width_st: view_width_st + self.view_size,
                                    pixel_height_st: pixel_height_st + self.crop_size,
                                    pixel_width_st: pixel_width_st + self.crop_size]

        # Augmentation:
        ground_truth = self.augmentation(ground_truth)

        # Blur
        blured = np.zeros(
            [self.view_size, self.view_size, self.crop_size // self.sr_rate, self.crop_size // self.sr_rate],
            dtype=ground_truth.dtype)
        for i in range(self.view_size):
            for j in range(self.view_size):
                blur = cv2.blur(ground_truth[i, j, :, :], (self.sr_rate, self.sr_rate))
                blured[i, j, :, :] = blur[self.sr_rate // 2::self.sr_rate, self.sr_rate // 2::self.sr_rate]
        ground_truth = ground_truth[np.newaxis, :]
        blured = blured[np.newaxis, :]
        ground_truth = torch.from_numpy((ground_truth).astype(np.float32))
        blured = torch.from_numpy((blured).astype(np.float32))
        ret = {
            'gt': ground_truth,
            'input': blured,
            'name': self.image_name_list[index]
        }
        return ret
