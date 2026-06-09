from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    Network contains an arbitrary 数量 hidden 层, ReLU nonlinearities,
    and a softmax 损失 函数. This 将 also implement dropout 并 batch/层
    归一化 as options. For a network 使用 L 层, architecture 将 be

    {affine - [batch/层 norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中 batch/层 归一化 并 dropout are optional 并 {...} block is
    repeated L - 1 times.

    Learnable 参数 are 存储 在 self.params 字典 并 将 be learned
    使用 Solver 类别.
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
        - hidden_dims: A list 的 integers giving size 的 each hidden 层.
        - 输入_dim: An integer giving size 的 输入.
        - num_类别: An integer giving 数量 类别 到 类别ify.
        - dropout_keep_ratio: Scalar between 0 并 1 giving dropout strength.
            If dropout_keep_ratio=1 然后 network 应该 不要使用 dropout at 所有.
        - 归一化: What type 的 归一化 network 应该 使用. Valid 值
            are "batchnorm", "层norm", or None 用于 no 归一化 (default).
        - reg: Scalar giving L2 正则化 strength.
        - weight_缩放: Scalar giving 标准差 用于 random
            initialization 的 权重.
        - dtype: A numpy 数据type object; 所有 computations 将 be performed 使用
            这个 数据type. float32 is faster but less accurate, so 你应该 使用
            float64 用于 numeric 梯度 checking.
        - seed: If not None, 然后 pass 这个 random seed 到 dropout 层.
            This 将 make dropout 层 deteriminstic so we 可以 梯度 check 模型.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化 参数 网络的, 存储所有值 在    #
        # self.params 字典. 存储 权重 并 偏置 用于 first 层 #
        # in W1 并 b1; 用于 second 层 使用 W2 并 b2, etc. Weights 应为 #
        # 初始化 来自 a normal distribution centered at 0 使用 standard       #
        # deviation equal 到 weight_缩放. Biases 应为 初始化 到 zero.   #
        #                                                                          #
        # When 使用 batch 归一化, 存储 缩放 并 平移 参数 用于 #
        # first 层 在 gamma1 并 beta1; 用于 second 层 使用 gamma2 并     #
        # beta2, etc. Scale 参数 应为 初始化 到 ones 并 平移     #
        # 参数 应为 初始化 到 zeros.                               #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # When 使用 dropout 我们需要 到 pass a dropout_param 字典 到 each
        # dropout 层 so 该 层 knows dropout 概率 并 模式
        # (训练 / 测试). 你可以 pass same dropout_param 到 each dropout 层.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch 归一化 我们需要 到 keep track 的 running 均值 and
        # 方差, so 我们需要 到 pass a special bn_param object 到 each batch
        # 归一化 层. 你应该 pass self.bn_params[0] 到 前向传播
        # of first batch 归一化 层, self.bn_params[1] 到 前向
        # pass 的 second batch 归一化 层, etc.
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
        - X: Array 的 输入 数据 的 形状 (N, d_1, ..., d_k)
        - y: Array 的 标签, 的 形状 (N,). y[i] gives 标签 用于 X[i].

        返回:
        If y is None, 然后 run a 测试时 前向传播 模型的 并 return:
        - 分数: Array 的 形状 (N, C) giving 分类 分数, 其中
            分数[i, c] is 分类 score 用于 X[i] 并 类别 c.

        If y is not None, 然后 run a 训练时 前向 并 反向传播 and
        return a tuple of:
        - 损失: Scalar 值 giving 损失
        - grads: Dictionary 使用 same keys as self.params, mapping parameter
            names 到 梯度 的 损失 使用 respect 到 those 参数.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set 训练/测试 模式 用于 batchnorm params 并 dropout param since they
        # behave differently during 训练 并 测试.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO：实现 前向传播 用于 fully connected net, 计算  #
        # 类别分数 用于 X 并 storing them 在 分数 变量.          #
        #                                                                          #
        # When 使用 dropout, you'll 需要 到 pass self.dropout_param 到 each       #
        # dropout 前向传播.                                                    #
        #                                                                          #
        # When 使用 batch 归一化, you'll 需要 到 pass self.bn_params[0] 到 #
        # 前向传播 用于 first batch 归一化 层, pass           #
        # self.bn_params[1] 到 前向传播 用于 second batch 归一化 #
        # 层, etc.                                                              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 如果是测试模式，则提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO：实现 反向传播 用于 fully connected net. 存储 #
        # 损失 在 损失 变量 并 梯度 在 grads 字典. 计算 #
        # 数据 损失 使用 softmax, 并 确保 该 grads[k] holds 梯度 #
        # for self.params[k]. 不要忘记 到 add L2 正则化!               #
        #                                                                          #
        # When 使用 batch/层 归一化, you don't 需要 到 regularize   #
        # 缩放 并 平移 参数.                                              #
        #                                                                          #
        # 注意： To ensure 该 your 实现 与参考实现匹配 并 能通过   #
        # 自动测试, 确保 该 your L2 正则化 包含 a 因子 #
        # of 0.5 到 简化表达式 用于 梯度.                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads
