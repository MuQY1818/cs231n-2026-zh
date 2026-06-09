from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel 数据 用于 images 并 several 特征 函数 该 可以 operate on
    single images, apply 所有 特征 函数 到 所有 images, concatenating the
    特征 vectors 用于 each image 并 storing 特征 用于 所有 images in
    a single 矩阵.

    输入:
    - imgs: N x H X W X C 数组 的 pixel 数据 用于 N images.
    - 特征_fns: List 的 k 特征 函数. ith 特征 函数 应该
      take as 输入 an H x W x D 数组 并 return a (one-维度al) 数组 of
      length F_i.
    - verbose: Boolean; if true, print progress.

    返回:
    An 数组 的 形状 (N, F_1 + ... + F_k) 其中 each column is concatenation
    of 所有 特征 用于 a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # 使用 first image 到 determine 特征维度
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now 该 we know 维度 的 特征, we 可以 分配 a single
    # big 数组 到 存储 所有 特征 as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract 特征 用于 rest 的 images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """将 RGB 图像转换为灰度图像

      Parameters:
        rgb : RGB image

      返回:
        gray : 灰度 image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """计算图像的方向梯度直方图（HOG）特征

         Modified 来自 skimage.特征.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.特征.hog

       Reference:
         Histograms 的 Oriented Gradients 用于 Human Detection
         Navneet Dalal 并 Bill Triggs, CVPR 2005

      Parameters:
        im : an 输入 灰度 or rgb image

      返回:
        feat: Histogram 的 Gradient (HOG) 特征

    """

    # 如果需要，将 RGB 转为灰度
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # 图像大小
    orientations = 9  # 梯度 bin 数量
    cx, cy = (8, 8)  # 每个 cell 的像素数

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # 计算 x 方向梯度
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # 计算 y 方向梯度
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # 梯度幅值
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # 梯度方向

    n_cellsx = int(np.floor(sx / cx))  # x 方向 cell 数量
    n_cellsy = int(np.floor(sy / cy))  # y 方向 cell 数量
    # 计算方向积分图
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # 为该方向创建新的积分图
        # 筛选该范围内的方向
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # 选择这些方向对应的幅值
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    计算 color histogram 用于 an image 使用 hue.

    输入:
    - im: H x W x C 数组 的 pixel 数据 用于 an RGB image.
    - nbin: 数量 histogram bins. (default: 10)
    - xmin: Minimum pixel 值 (default: 0)
    - xmax: Maximum pixel 值 (default: 255)
    - 归一化后的: Whether 到 normalize histogram (default: True)

    返回:
      1D vector 的 length nbin giving color histogram 在 hue 的 the
      输入 image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # 返回直方图
    return imhist


# ~~START DELETE~~
# These are some other 特征 该 we implemented 到 play 约为, but 不是
# distributing 到 students.
def color_histogram(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """计算图像的颜色直方图特征

      Parameters:
        im : a numpy 数组 的 灰度 or rgb image
        nbin : 数量 histogram bins (default: 10)
        xmin : minimum pixel 值 (default: 0)
        xmax : maximum pixel 值 (deafult: 255)
        归一化后的 : bool flag 到 normalize histogram

      返回:
        feat : color histogram 特征

    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    # 灰度图像
    if ndim == 2:
        imhist, bin_edges = np.histogram(im, bins=bins, density=normalized)
        return imhist
    # RGB 图像
    elif ndim == 3:
        color_hist = np.array([])
        # 遍历三个颜色通道
        for k in range(3):
            # 计算归一化直方图
            imhist, bin_edges = np.histogram(im[:, :, k], bins=bins, density=normalized)
            imhist = imhist * np.diff(bin_edges)
            # 拼接直方图
            color_hist = np.concatenate((color_hist, imhist))
        # 返回直方图
        return color_hist
    # 未知图像类型
    return np.array([])


def color_histogram_spatial(img, levels=3, nbin=4):
    """
    Color histogram 在 a pyramid.
    """
    feats = []

    for level in range(1, levels + 1):
        chunks = np.array_split(img, level, axis=0)
        chunks = [np.array_split(chunk, level, axis=1) for chunk in chunks]
        for x in chunks:
            for chunk in x:
                feats.append(color_histogram_cross(chunk, nbin=nbin))

    return np.hstack(feats)


def color_histogram_cross(img, nbin=5, normalized=True):
    """
    RGB color histogram 其中 our bins are 3 维度al.
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
