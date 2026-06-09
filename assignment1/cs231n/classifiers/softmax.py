from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax 损失函数的朴素实现（使用循环）。

    输入维度为 D，共有 C 个类别；本函数在包含 N 个样本的 minibatch 上操作。

    输入:
    - W: 形状为 (D, C) 的 numpy 数组，包含权重。
    - X: 形状为 (N, D) 的 numpy 数组，包含一个 minibatch 的数据。
    - y: 形状为 (N,) 的 numpy 数组，包含训练标签；
      y[i] = c 表示 X[i] 的标签为 c，其中 0 <= c < C。
    - reg: (float) 正则化强度。

    返回一个 tuple：
    - 单个 float 形式的损失。
    - 关于权重 W 的梯度；形状与 W 相同。
    """
    # 将损失和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # 以数值稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # 归一化
        logp = np.log(p)

        loss -= logp[y[i]]  # 负对数概率即为损失


    # 对损失取平均并加上正则化项。
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算损失函数的梯度，并将其存入 dW。与其先计算损失再计算导数，             #
    # 更简单的做法可能是在计算损失的同时计算导数。                              #
    # 因此你可能需要修改上面部分代码来计算梯度。                                #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax 损失函数的向量化版本。

    输入和输出与 softmax_loss_naive 相同。
    """
    # 将损失和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # 实现 softmax 损失的向量化版本，并将结果存入 loss。                       #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # 实现 softmax 损失梯度的向量化版本，并将结果存入 dW。                     #
    #                                                                           #
    # 提示：与其从头计算梯度，复用你计算损失时得到的中间值可能更容易。          #
    #############################################################################


    return loss, dW
