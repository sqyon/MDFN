import importlib

import imageio
import numpy as np
import os
import skimage.color as color
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_loss(loss_name, loss_config):
    """Load loss form loss_name."""
    loss_lib = importlib.import_module(f'loss.{loss_name}')
    loss_func = getattr(loss_lib, 'get_loss')
    loss = loss_func(loss_config)
    return loss


def load_model(model_name, network_config):
    """Load model form model_name."""
    model_lib = importlib.import_module(f'model.{model_name}')
    model_func = getattr(model_lib, 'get_model')
    model = model_func(network_config)
    return model


def load_single_config(config_name):
    """Load single configuration file, and return config dictionary."""
    config_lib = importlib.import_module(f'config.{config_name}')
    config_func = getattr(config_lib, 'get_config')
    config_dict = config_func()
    return config_dict


def dfs_config(config, default):
    if type(config) != dict or type(default) != dict:
        return config
    ret = {}
    for key in default:
        if key not in config:
            ret[key] = default[key]
        else:
            ret[key] = dfs_config(config[key], default[key])
    return ret


def load_configs(config_name, default_config_name):
    """Load experiment configuration and merge it with default configuration."""
    default_config = load_single_config(default_config_name)
    config = load_single_config(config_name)
    configs = dfs_config(config, default_config)
    return configs


def set_gpu(gpu_config):
    """Set CUDA environment, return the number of used GPUs."""
    gpu = gpu_config['visible_gpu']
    if gpu != -1:
        assert type(gpu) is list, "gpu_config['visible_gpu'] must be -1 or a list of integers."
        gpu_str = ''
        for i in gpu:
            assert type(i) is int, "gpu_config['visible_gpu'] must be -1 or a list of integers."
            gpu_str += f'{i},'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str[:-1]
    return torch.cuda.device_count()


def split_test_image(x, shave=10):
    """Split test image into small size to avoid OOM error."""
    b, c, u, v, h, w = x.size()
    h_mid, w_mid = h // 2, w // 2
    h_size, w_size = h_mid + shave, w_mid + shave
    ret = torch.zeros([b * 4, c, u, v, h_size, w_size], dtype=x.dtype, device=x.device)
    ret[:b] = x[:, :, :, :, :h_size, :w_size]
    ret[b:2 * b] = x[:, :, :, :, :h_size, w - w_size:]
    ret[2 * b:3 * b] = x[:, :, :, :, h - h_size:, :w]
    ret[3 * b:] = x[:, :, :, :, h - h_size:, w - w_size:]
    return ret


def merge_test_image(x, shape, shave=10):
    """Merge splited test image into small size."""
    b, c, u, v, h, w = shape
    h_mid, w_mid = h - shave, w - shave
    h_size, w_size = h_mid + shave, w_mid + shave
    ret = torch.zeros([b // 4, c, u, v, h_size, w_size], dtype=x.dtype, device=x.device)


class LossStatistics:
    """Statistic loss."""

    def __init__(self):
        self.loss = {}

    def append(self, x):
        for i in x:
            if i not in self.loss:
                self.loss[i] = []
            self.loss[i].append(x[i])

    def eval(self):
        average = {}
        for i in self.loss:
            average[i] = float(sum(self.loss[i]) / len(self.loss[i]))
        return average


class ImageMetrics:
    """Measure predicted image with ground truth."""

    def __init__(self):
        self.psnr_list = []
        self.ssim_list = []

    def append(self, x):
        ground_truth = x['gt']
        predict = x['output']
        shape = list(ground_truth.shape)
        N, C, Hv, Wv, Hp, Wp = shape
        gt = ground_truth.permute(0, 2, 3, 4, 5, 1)
        gt = gt.reshape(N * Hv * Wv, Hp, Wp, C)
        y_hat = predict.permute(0, 2, 3, 4, 5, 1)
        y_hat = y_hat.reshape(N * Hv * Wv, Hp, Wp, C)
        gt = gt.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        y_hat = np.clip(y_hat, 16 / 255, 235 / 255)
        for i in range(gt.shape[0]):
            self.psnr_list.append(peak_signal_noise_ratio(gt[i], y_hat[i]))
            self.ssim_list.append(structural_similarity(gt[i], y_hat[i], multichannel=True))

    def eval(self):
        psnr = sum(self.psnr_list) / len(self.psnr_list)
        ssim = sum(self.ssim_list) / len(self.ssim_list)
        ret = {
            'psnr': psnr,
            'ssim': ssim
        }
        return ret


def save_light_field_images(file_path, img, mode='YCbCr'):
    """Save Light Field Images as png.
    :param file_path: If there are multiple images, the file_path will end with suffix.
    :param img: Must be a torch.Tensor or np.array. The shape of img must be
                    (batch, channel, view, view, h, w) or (channel, view, view, h, w).
    :param mode: Image color mode, must be 'YCbCr', 'RGB' or 'Gray'.
    """

    def _save_single_lf_image(save_path, img, mode):
        c, vh, vw, h, w = list(img.shape)
        img = np.transpose(img, [1, 3, 2, 4, 0])
        img = img.reshape([vh * h, vw * w, c])
        if np.max(img) < 1.1:
            img *= 255
        if mode == 'ycbcr':
            img = color.ycbcr2rgb(img)
        imageio.imwrite(save_path, img)

    mode = mode.lower()
    assert mode in ['ycbcr', 'rgb', 'gray']
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()
    assert type(img) is np.ndarray, f'Assert img is torch.Tensor of np.ndarray, but got {type(img)}'
    dir = file_path[:file_path.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)
    if len(img.shape) == 4:
        if file_path[:-4] == '.png':
            save_name = f'{file_path[:-4]}.png'
        else:
            save_name = file_path
        _save_single_lf_image(save_name, img, mode)
        return
    assert len(
        img.shape) == 5, "The shape of LF image must be (batch, channel, view, view, h, w) or (channel, view, view, h, w)"
    for i in range(img.shape[0]):
        if file_path[:-4] == '.png':
            save_name = f'{file_path[:-4]}_{i}.png'
        else:
            save_name = f'{file_path}_{i}.png'
        _save_single_lf_image(save_name, img, mode)
