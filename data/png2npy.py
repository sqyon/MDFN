from multiprocessing.pool import Pool

import imageio
import numpy as np
import os
import skimage.color as color


def process_single_image(param):
    png, png_path, npy_path = param
    image_path = f'{png_path}/{png}'
    rgb = imageio.imread(image_path)
    ycbcr = color.rgb2ycbcr(rgb[:, :, :3])
    height, width = ycbcr.shape[0] // 14, ycbcr.shape[1] // 14
    ycbcr = ycbcr.reshape([height, 14, width, 14, 3])
    ycbcr = ycbcr[15:-15, 3:-3, 15:-15, 3:-3, :].astype(np.float32) / 255.0
    ycbcr = np.transpose(ycbcr, [1, 3, 0, 2, 4])
    np.save(f'{npy_path}/{png[:-4]}.npy', ycbcr)
    print(f'{ycbcr.shape} @ {npy_path}/{png[:-4]}.npy')
    return ycbcr


def convert(png_path, npy_path):
    """ Convert RGB png image into YCbCr npy file.
            The shape of saved file is (8, 8, h, w, 3). Range of saved file is [0, 1].
    """
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    png_names = os.listdir(png_path)
    n = len(png_names)
    png_path = [png_path] * n
    npy_path = [npy_path] * n
    param = zip(png_names, png_path, npy_path)
    with Pool(1) as p:
        p.map(process_single_image, param)


if __name__ == '__main__':
    convert('./data/LFdataset/rawdata/general/raw', './data/LFdataset/validation/ycbcr')
