from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train 这个 linear 分类器 使用 stochastic 梯度 descent.

        输入:
        - X: A numpy 数组 的 形状 (N, D) containing 训练数据; there are N
          训练 样本 each 的 维度 D.
        - y: A numpy 数组 的 形状 (N,) containing 训练 标签; y[i] = c
          均值 该 X[i] has 标签 0 <= c < C 用于 C 类别.
        - learning_rate: (float) 学习率 用于 optimization.
        - reg: (float) 正则化 strength.
        - num_iters: (integer) 数量 steps 到 take when optimizing
        - batch_size: (integer) 数量 训练 样本 到 使用 at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing 值 的 损失 函数 at each 训练 iteration.
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes 值 0...K-1 其中 K is 数量 类别
        if self.W is None:
            # lazily 初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic 梯度 descent 到 optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements 来自 训练数据 并 their           #
            # corresponding 标签 到 使用 在 这个 round 的 梯度 descent.        #
            # 将数据存储 在 X_batch 并 their corresponding 标签 在           #
            # y_batch; after sampling X_batch 应该 have 形状 (batch_size, dim)   #
            # and y_batch 应该 have 形状 (batch_size,)                           #
            #                                                                       #
            # 提示： 使用 np.random.choice 到 generate indices. Sampling 使用         #
            # replacement is 快于 sampling 无放回.              #
            #########################################################################


            # 计算损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # Update 权重 使用 梯度 并 学习率.          #
            #########################################################################


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用 训练ed 权重 的 这个 linear 分类器 到 predict 标签 for
        数据 点.

        输入:
        - X: A numpy 数组 的 形状 (N, D) containing 训练数据; there are N
          训练 样本 each 的 维度 D.

        返回:
        - y_pred: Predicted 标签 用于 数据 在 X. y_pred is a 1-维度al
          数组 的 length N, 并 each element is an integer giving 预测的
          类别.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement 这个 method. 存储 预测的 标签 在 y_pred.            #
        ###########################################################################

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算 损失 函数 并 its derivative.
        Sub类别 将 在ride 这个.

        输入:
        - X_batch: A numpy 数组 的 形状 (N, D) containing a minibatch 的 N
          数据 点; each 点 has 维度 D.
        - y_batch: A numpy 数组 的 形状 (N,) containing 标签 用于 minibatch.
        - reg: (float) 正则化 strength.

        返回: A tuple containing:
        - 损失 as a single float
        - 梯度 使用 respect 到 self.W; an 数组 的 same 形状 as W
        """
        pass

    def save(self, fname):
      """保存模型参数。"""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = {"W": self.W}
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """加载模型参数。"""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.W = params["W"]
        print(fname, "loaded.")
        return True


class LinearSVM(LinearClassifier):
    """ 使用 Multi类别 SVM 损失函数的子类 """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ 使用 Softmax + Cross-entropy 损失函数的子类 """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
