from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    给定图像像素数据和若干可作用于单张图像的特征函数，
    将所有特征函数应用到所有图像上，拼接每张图像的特征向量，
    并把所有图像的特征存储在一个矩阵中。

    输入:
    - imgs: N x H x W x C 数组，表示 N 张图像的像素数据。
    - feature_fns: k 个特征函数组成的列表。第 i 个特征函数应接收
      H x W x D 数组作为输入，并返回长度为 F_i 的一维数组。
    - verbose: 布尔值；若为 true，则打印进度。

    返回:
    形状为 (N, F_1 + ... + F_k) 的数组，其中每一行是单张图像所有特征的拼接。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # 使用第一张图像确定特征维度。
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # 现在已知特征维度，可以分配一个大数组来存储所有特征。
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # 提取其余图像的特征。
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

      参数:
        rgb : RGB 图像

      返回:
        gray : 灰度图像

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """计算图像的方向梯度直方图（HOG）特征

         修改自 skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.特征.hog

       参考:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      参数:
        im : 输入灰度图或 RGB 图像

      返回:
        feat: HOG 特征

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
    使用 hue 计算图像的颜色直方图。

    输入:
    - im: H x W x C 数组，表示 RGB 图像的像素数据。
    - nbin: histogram bin 数量（默认 10）。
    - xmin: 最小像素值（默认 0）。
    - xmax: 最大像素值（默认 255）。
    - normalized: 是否归一化 histogram（默认 True）。

    返回:
      长度为 nbin 的一维向量，表示输入图像 hue 通道上的颜色直方图。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # 返回直方图
    return imhist


# ~~START DELETE~~
# 下面是一些我们实现来做实验的其他特征，但不会分发给学生。
def color_histogram(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """计算图像的颜色直方图特征

      参数:
        im : 灰度或 RGB 图像的 numpy 数组
        nbin : histogram bin 数量（默认 10）
        xmin : 最小像素值（默认 0）
        xmax : 最大像素值（默认 255）
        normalized : 是否归一化 histogram 的布尔标志

      返回:
        feat : 颜色直方图特征

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
    金字塔上的颜色直方图。
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
    RGB 颜色直方图，其中 bin 是三维的。
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
