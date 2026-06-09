from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    一个使用 ReLU 非线性和 softmax 损失的两层全连接神经网络，
    采用模块化层设计。我们假设输入维度为 D，隐藏层维度为 H，
    并在 C 个类别上进行分类。

    网络结构应为 affine - relu - affine - softmax。

    注意：这个类不实现梯度下降；它会与单独的 Solver 对象配合，
    由 Solver 负责执行优化过程。

    模型的可学习参数存储在 self.params 字典中；
    该字典将参数名映射到 numpy 数组。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        初始化一个新网络。

        输入:
        - input_dim: 整数，表示输入大小。
        - hidden_dim: 整数，表示隐藏层大小。
        - num_classes: 整数，表示分类类别数。
        - weight_scale: 标量，表示随机初始化权重时使用的标准差。
        - reg: 标量，表示 L2 正则化强度。
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: 初始化两层网络的权重和偏置。权重应从均值为 0.0、标准差为       #
        # weight_scale 的高斯分布中初始化，偏置应初始化为 0。所有权重和偏置     #
        # 都应存储在 self.params 字典中：第一层权重和偏置使用键 'W1'、'b1'，  #
        # 第二层权重和偏置使用键 'W2'、'b2'。                                  #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        计算一个 minibatch 数据的损失和梯度。

        输入:
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)。
        - y: 标签数组，形状为 (N,)。y[i] 是 X[i] 的标签。

        返回:
        如果 y 为 None，则执行测试时前向传播并返回：
        - scores: 形状为 (N, C) 的数组，给出分类分数；
          scores[i, c] 是样本 X[i] 属于类别 c 的分数。

        如果 y 不为 None，则执行训练时前向传播和反向传播，并返回一个 tuple：
        - loss: 标量，表示损失。
        - grads: 字典，键与 self.params 相同，将参数名映射到对应损失梯度。
        """
        scores = None
        ############################################################################
        # TODO：实现两层网络的前向传播，计算 X 的类别分数并存入 scores 变量。    #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 如果 y 为 None，则处于测试模式，直接返回 scores。
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO：实现两层网络的反向传播。将损失存入 loss 变量，梯度存入 grads     #
        # 字典。使用 softmax 计算数据损失，并确保 grads[k] 保存的是             #
        # self.params[k] 对应的梯度。不要忘记加入 L2 正则化。                   #
        #                                                                          #
        # 注意：为了与参考实现匹配并通过自动测试，L2 正则化项应包含 0.5 因子，   #
        # 这样可以简化梯度表达式。                                               #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """保存模型参数。"""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """多层全连接神经网络类。

    网络可以包含任意数量的隐藏层、ReLU 非线性和一个 softmax 损失函数。
    本类还会可选实现 dropout 以及 batch/layer normalization。
    对于一个 L 层网络，其结构为：

    {affine - [batch/层 norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中 batch/layer normalization 和 dropout 都是可选的，
    {...} 这个 block 会重复 L - 1 次。

    可学习参数存储在 self.params 字典中，并将由 Solver 类学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化新的 FullyConnectedNet。

        输入:
        - hidden_dims: 整数列表，给出每个隐藏层的大小。
        - input_dim: 整数，给出输入大小。
        - num_classes: 整数，给出分类类别数。
        - dropout_keep_ratio: 0 到 1 之间的标量，表示 dropout 保留比例。
          如果 dropout_keep_ratio=1，则网络不使用 dropout。
        - normalization: 网络使用的归一化类型；合法值为 "batchnorm"、
          "layernorm" 或 None（默认表示不使用归一化）。
        - reg: 标量，表示 L2 正则化强度。
        - weight_scale: 标量，表示随机初始化权重时使用的标准差。
        - dtype: numpy 数据类型；所有计算都会使用该数据类型。
          float32 更快但不够精确，因此做数值梯度检查时应使用 float64。
        - seed: 若不为 None，则将这个随机种子传给 dropout 层。
          这会让 dropout 层具有确定性，从而便于做梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络参数，并将所有值存储在 self.params 字典中。第一层      #
        # 的权重和偏置存为 W1 和 b1，第二层存为 W2 和 b2，依此类推。权重应从    #
        # 均值为 0、标准差为 weight_scale 的正态分布中初始化，偏置初始化为 0。   #
        #                                                                          #
        # 使用 batch normalization 时，第一层的缩放和平移参数存为 gamma1 和     #
        # beta1，第二层存为 gamma2 和 beta2，依此类推。缩放参数初始化为 1，      #
        # 平移参数初始化为 0。                                                    #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 使用 dropout 时，需要向每个 dropout 层传入 dropout_param 字典，
        # 让层知道 dropout 概率和当前模式（训练/测试）。
        # 可以把同一个 dropout_param 传给每个 dropout 层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用 batch normalization 时，需要跟踪 running mean 和 running variance，
        # 因此要向每个 batch normalization 层传入一个特殊的 bn_param 对象。
        # 第一层 batch normalization 的前向传播应使用 self.bn_params[0]，
        # 第二层使用 self.bn_params[1]，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。
        
        输入:
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)。
        - y: 标签数组，形状为 (N,)。y[i] 是 X[i] 的标签。

        返回:
        如果 y 为 None，则执行测试时前向传播并返回：
        - scores: 形状为 (N, C) 的数组，给出分类分数；
          scores[i, c] 是样本 X[i] 属于类别 c 的分数。

        如果 y 不为 None，则执行训练时前向传播和反向传播，并返回一个 tuple：
        - loss: 标量，表示损失。
        - grads: 字典，键与 self.params 相同，将参数名映射到对应损失梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 设置 batchnorm 参数和 dropout 参数的训练/测试模式；
        # 它们在训练和测试时行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO：实现全连接网络的前向传播，计算 X 的类别分数并存入 scores 变量。  #
        #                                                                          #
        # 使用 dropout 时，需要把 self.dropout_param 传给每个 dropout 前向传播。 #
        #                                                                          #
        # 使用 batch normalization 时，第一层 batch normalization 的前向传播要    #
        # 传入 self.bn_params[0]，第二层传入 self.bn_params[1]，依此类推。        #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 如果是测试模式，则提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO：实现全连接网络的反向传播。将损失存入 loss 变量，梯度存入 grads   #
        # 字典。使用 softmax 计算数据损失，并确保 grads[k] 保存的是             #
        # self.params[k] 对应的梯度。不要忘记加入 L2 正则化。                   #
        #                                                                          #
        # 使用 batch/layer normalization 时，不需要正则化缩放和平移参数。        #
        #                                                                          #
        # 注意：为了与参考实现匹配并通过自动测试，L2 正则化项应包含 0.5 因子，   #
        # 这样可以简化梯度表达式。                                               #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """保存模型参数。"""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True
