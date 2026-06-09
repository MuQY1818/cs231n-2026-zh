from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN 分类器 使用 L2 距离 """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train 分类器. For k-最近邻 这个 is just
        memorizing 训练数据.

        输入:
        - X: A numpy 数组 的 形状 (num_训练, D) containing 训练数据
          consisting 的 num_训练 样本 each 的 维度 D.
        - y: A numpy 数组 的 形状 (N,) containing 训练 标签, 其中
             y[i] is 标签 用于 X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict 标签 用于 测试 数据 使用 这个 分类器.

        输入:
        - X: A numpy 数组 的 形状 (num_测试, D) containing 测试 数据 consisting
             of num_测试 样本 each 的 维度 D.
        - k: 数量 最近邻 该 vote 用于 预测的 标签.
        - num_loops: Determines which 实现 到 使用 到 计算 距离
          between 训练点 并 测试点.

        返回:
        - y: A numpy 数组 的 形状 (num_测试,) containing 预测的 标签 用于 the
          测试 数据, 其中 y[i] is 预测的 标签 用于 测试点 X[i].
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
        计算 距离 between each 测试点 在 X 并 each 训练点
        in self.X_训练 使用 a nested loop 在 both 训练数据 并 the
        测试 数据.

        输入:
        - X: A numpy 数组 的 形状 (num_测试, D) containing 测试 数据.

        返回:
        - dists: A numpy 数组 的 形状 (num_测试, num_训练) 其中 dists[i, j]
          is Euclidean 距离 between ith 测试点 并 jth 训练
          点.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算 l2 距离 between ith 测试点 并 jth    #
                # 训练点, 并 存储 结果 在 dists[i, j]. 你应该   #
                # 不要使用 a loop 在 维度, 也不要使用 np.linalg.norm().          #
                #####################################################################
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        计算 距离 between each 测试点 在 X 并 each 训练点
        in self.X_训练 使用 a single loop 在 测试 数据.

        输入 / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算 l2 距离 between ith 测试点 并 所有 训练 #
            # 点, 并 存储 结果 在 dists[i, :].                        #
            # 不要 使用 np.linalg.norm().                                        #
            #######################################################################
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        计算 距离 between each 测试点 在 X 并 each 训练点
        in self.X_训练 使用 no explicit loops.

        输入 / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # 计算 l2 距离 between 所有 测试点 并 所有 训练      #
        # 点 不使用 使用 any explicit loops, 并 存储 结果 在      #
        # dists.                                                                #
        #                                                                       #
        # 你应该 implement 这个 函数 使用 only 基础数组操作; #
        # in particular 你应该 不要使用 scipy 函数,                #
        # 也不要使用 np.linalg.norm().                                             #
        #                                                                       #
        # 提示： Try 到 formulate l2 距离 使用 矩阵乘法    #
        #       并 two broadcast 求和.                                         #
        #########################################################################

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a 矩阵 的 距离 between 测试点 并 训练点,
        predict a 标签 用于 each 测试点.

        输入:
        - dists: A numpy 数组 的 形状 (num_测试, num_训练) 其中 dists[i, j]
          gives 距离 betwen ith 测试点 并 jth 训练点.

        返回:
        - y: A numpy 数组 的 形状 (num_测试,) containing 预测的 标签 用于 the
          测试 数据, 其中 y[i] is 预测的 标签 用于 测试点 X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list 的 length k storing 标签 的 k 最近邻 to
            # ith 测试点.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用 距离矩阵 到 找到 k 最近邻 的 ith    #
            # 测试点, 并 使用 self.y_训练 到 找到 标签 的 这些       #
            # 邻居. 将这些标签存储 在 closest_y.                           #
            # 提示： Look up 函数 numpy.argsort.                             #
            #########################################################################


            #########################################################################
            # TODO:                                                                 #
            # Now 该 you have found 标签 的 k 最近邻, you    #
            # 需要 到 找到 most common 标签 在 list closest_y 的 标签.   #
            # 将该标签存储 在 y_pred[i]. Break ties by 选择 更小     #
            # 标签.                                                                #
            #########################################################################


        return y_pred
