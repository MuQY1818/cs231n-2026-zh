"""工具函数 使用 用于 查看和处理图像."""

import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from imageio import imread
from PIL import Image



def blur_image(X):
    """
    A very gentle image blurring operation, 被使用 as a regularizer for
    image generation.

    输入:
    - X: Image 数据 的 形状 (N, 3, H, W)

    返回:
    - X_blur: Blurred version 的 X, 的 形状 (N, 3, H, W)
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
    """预处理图像 用于 squeezenet.

    Subtracts pixel 均值 并 divides by 标准差.
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """撤销预处理 on an image 并 convert back 到 uint8."""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image 来自 a URL. 返回 a numpy 数组 使用 pixel 数据.
    We write image 到 a temporary file 然后 read it back. Kinda gross.
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
    """从磁盘加载并 resize 图像.

    输入:
    - filename: path 到 file
    - size: size 的 最短 维度 after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: width, height 值 are currently flipped here, 并 we 应该
        # change resampling method 到 BILINEAR 到 match torch 实现
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
