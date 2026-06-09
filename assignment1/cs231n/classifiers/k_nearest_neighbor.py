from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """使用 L2 距离的 kNN 分类器。"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。对于 k 最近邻来说，训练过程只是记住训练数据。

        输入:
        - X: 形状为 (num_train, D) 的 numpy 数组，包含训练数据；
          其中有 num_train 个样本，每个样本维度为 D。
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签；
          y[i] 是 X[i] 的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用该分类器预测测试数据的标签。

        输入:
        - X: 形状为 (num_test, D) 的 numpy 数组，包含测试数据；
          其中有 num_test 个样本，每个样本维度为 D。
        - k: 用于投票预测标签的最近邻数量。
        - num_loops: 决定用哪一种实现来计算训练点和测试点之间的距离。

        返回:
        - y: 形状为 (num_test,) 的 numpy 数组，包含测试数据的预测标签；
          y[i] 是测试点 X[i] 的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        使用双重循环，计算 X 中每个测试点与 self.X_train 中每个训练点的距离。

        输入:
        - X: 形状为 (num_test, D) 的 numpy 数组，包含测试数据。

        返回:
        - dists: 形状为 (num_test, num_train) 的 numpy 数组，
          其中 dists[i, j] 是第 i 个测试点和第 j 个训练点之间的欧氏距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算第 i 个测试点和第 j 个训练点之间的 L2 距离，                 #
                # 并将结果存入 dists[i, j]。你不应该在维度上再写循环，             #
                # 也不能使用 np.linalg.norm()。                                    #
                #####################################################################
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        使用一个测试数据循环，计算 X 中每个测试点与 self.X_train 中每个训练点的距离。

        输入 / 输出: 与 compute_distances_two_loops 相同。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算第 i 个测试点与所有训练点之间的 L2 距离，                       #
            # 并将结果存入 dists[i, :]。不要使用 np.linalg.norm()。               #
            #######################################################################
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        不使用显式循环，计算 X 中每个测试点与 self.X_train 中每个训练点的距离。

        输入 / 输出: 与 compute_distances_two_loops 相同。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # 不使用任何显式循环，计算所有测试点和所有训练点之间的 L2 距离，         #
        # 并将结果存入 dists。                                                   #
        #                                                                       #
        # 你应该只使用基础数组操作来实现这个函数；尤其不能使用 scipy 函数，       #
        # 也不能使用 np.linalg.norm()。                                           #
        #                                                                       #
        # 提示：尝试用矩阵乘法和两个 broadcast 求和来表示 L2 距离。              #
        #########################################################################

        return dists

    def predict_labels(self, dists, k=1):
        """
        给定测试点和训练点之间的距离矩阵，预测每个测试点的标签。

        输入:
        - dists: 形状为 (num_test, num_train) 的 numpy 数组，
          其中 dists[i, j] 表示第 i 个测试点和第 j 个训练点之间的距离。

        返回:
        - y: 形状为 (num_test,) 的 numpy 数组，包含测试数据的预测标签；
          y[i] 是测试点 X[i] 的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 长度为 k 的列表，用来存储第 i 个测试点的 k 个最近邻标签。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用距离矩阵找到第 i 个测试点的 k 个最近邻，                         #
            # 再用 self.y_train 找到这些邻居的标签。将这些标签存入 closest_y。      #
            # 提示：查阅 numpy.argsort 函数。                                       #
            #########################################################################


            #########################################################################
            # TODO:                                                                 #
            # 现在你已经找到了 k 个最近邻的标签，需要在 closest_y 中找出最常见的   #
            # 标签，并将其存入 y_pred[i]。如果出现平票，选择数值更小的标签。       #
            #########################################################################


        return y_pred
