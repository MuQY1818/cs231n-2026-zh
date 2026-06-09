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
        使用随机梯度下降训练这个线性分类器。

        输入:
        - X: 形状为 (N, D) 的 numpy 数组，包含训练数据；
          其中有 N 个训练样本，每个样本维度为 D。
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签；
          y[i] = c 表示 X[i] 的标签为 c，且 0 <= c < C。
        - learning_rate: (float) 优化时使用的学习率。
        - reg: (float) 正则化强度。
        - num_iters: (integer) 优化时迭代的步数。
        - batch_size: (integer) 每一步使用的训练样本数量。
        - verbose: (boolean) 若为 true，则在优化过程中打印进度。

        输出:
        一个列表，包含每次训练迭代时的损失值。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # 假设 y 的取值为 0...K-1，其中 K 是类别数。
        if self.W is None:
            # 懒初始化 W。
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 运行随机梯度下降来优化 W。
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # 从训练数据中采样 batch_size 个元素及其对应标签，                     #
            # 用于这一轮梯度下降。将采样数据存入 X_batch，                         #
            # 将对应标签存入 y_batch。采样后，X_batch 的形状应为                  #
            # (batch_size, dim)，y_batch 的形状应为 (batch_size,)。                #
            #                                                                       #
            # 提示：使用 np.random.choice 生成索引。有放回采样比无放回采样更快。   #
            #########################################################################


            # 计算损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # 使用梯度和学习率更新权重。                                           #
            #########################################################################


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用这个线性分类器训练得到的权重，为数据点预测标签。

        输入:
        - X: 形状为 (N, D) 的 numpy 数组，包含 N 个数据样本；
          每个样本维度为 D。

        返回:
        - y_pred: X 中各个数据点的预测标签。y_pred 是长度为 N 的一维数组，
          每个元素是一个整数，表示预测类别。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # 实现这个方法，并将预测标签存入 y_pred。                                #
        ###########################################################################

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数。子类会覆写这个方法。

        输入:
        - X_batch: 形状为 (N, D) 的 numpy 数组，包含一个 minibatch；
          其中有 N 个数据点，每个点维度为 D。
        - y_batch: 形状为 (N,) 的 numpy 数组，包含该 minibatch 的标签。
        - reg: (float) 正则化强度。

        返回: 一个 tuple，包含：
        - 单个 float 形式的损失。
        - 关于 self.W 的梯度；形状与 W 相同。
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
    """使用 Multiclass SVM 损失函数的子类。"""

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """使用 Softmax + Cross-entropy 损失函数的子类。"""

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
