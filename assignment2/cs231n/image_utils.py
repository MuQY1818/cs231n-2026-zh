from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from imageio import imread
from PIL import Image

"""
Utility functions used for viewing and processing images.
"""


def blur_image(X):
    """
    一个非常轻微的图像模糊操作，用作 image generation 的 regularizer。

    输入:
    - X: 图像数据，形状为 (N, 3, H, W)

    返回:
    - X_blur: X 的模糊版本，形状为 (N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """为 squeezenet 预处理图像。

    减去像素均值，并除以标准差。
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """撤销图像预处理，并转换回 uint8。"""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    从 URL 读取图像，返回包含像素数据的 numpy 数组。我们会先把图像写入临时文件，
    再读回来。这个做法有点粗糙。
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, "wb") as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print("URL Error: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP Error: ", e.code, url)


def load_image(filename, size=None):
    """从磁盘加载图像并 resize。

    输入:
    - filename: 文件路径
    - size: rescale 后最短维度的大小
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: 这里 width 和 height 的值目前是反的；还应把 resampling method
        # 改为 BILINEAR，以匹配 torch 实现。
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
