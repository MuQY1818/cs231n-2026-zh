from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax 损失 函数, naive 实现 (使用 loops)

    输入 have 维度 D, there are C 类别, 并 we operate on minibatches
    of N 样本.

    输入:
    - W: A numpy 数组 的 形状 (D, C) containing 权重.
    - X: A numpy 数组 的 形状 (N, D) containing a minibatch 的 数据.
    - y: A numpy 数组 的 形状 (N,) containing 训练 标签; y[i] = c 均值
      该 X[i] has 标签 c, 其中 0 <= c < C.
    - reg: (float) 正则化 strength

    返回 a tuple of:
    - 损失 as single float
    - 梯度 使用 respect 到 权重 W; an 数组 的 same 形状 as W
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


    # 归一化 hinge 损失 加正则化
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算 梯度 的 损失 函数 并 存储 it dW.                #
    # Rather 该 first 计算 损失 并 然后 计算 derivative,   #
    # it may be simpler 到 计算 derivative at same time 该     #
    # 损失 is being 计算得到的. As a 结果 你可以 需要 到 modify some 的    #
    # code above 到 计算 梯度.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax 损失 函数, 向量化版本.

    输入 并 输出 are same as softmax_损失_naive.
    """
    # 将损失和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a 向量化版本 的 softmax 损失, storing           #
    # 结果 在 损失.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a 向量化版本 的 梯度 用于 softmax            #
    # 损失, 将结果存储 在 dW.                                           #
    #                                                                           #
    # 提示： Instead 的 计算 梯度 来自 scratch, it may be easier    #
    # 复用 some 的 中间值 该 you 使用 到 计算     #
    # 损失.                                                                     #
    #############################################################################


    return loss, dW
