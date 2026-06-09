from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-层 convolutional network 使用 following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    network operates on minibatches 的 数据 该 have 形状 (N, C, H, W)
    consisting 的 N images, each 使用 height H 并 width W 并 使用 C 输入
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        初始化 a new network.

        输入:
        - 输入_dim: Tuple (C, H, W) giving size 的 输入 数据
        - num_filters: 数量 filters 到 使用 在 convolutional 层
        - filter_size: Width/height 的 filters 到 使用 在 convolutional 层
        - hidden_dim: 数量 units 到 使用 在 fully-connected hidden 层
        - num_类别: 数量 分数 到 produce 来自 final affine 层.
        - weight_缩放: Scalar giving 标准差 用于 random initialization
          of 权重.
        - reg: Scalar giving L2 正则化 strength
        - dtype: numpy 数据type 到 使用 用于 computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: 初始化 权重 并 偏置 用于 three-层 convolutional    #
        # network. Weights 应为 初始化 来自 a Gaussian centered at 0.0   #
        # 使用 标准差 equal 到 weight_缩放; 偏置 应为          #
        # 初始化 到 zero. All 权重 并 偏置 应为 存储 在      #
        #  字典 self.params. 存储 权重 并 偏置 用于 convolutional  #
        # 层 使用 keys 'W1' 并 'b1'; 使用 keys 'W2' 并 'b2' 用于       #
        # 权重 并 偏置 的 hidden affine 层, 并 keys 'W3' 并 'b3'    #
        # for 权重 并 偏置 的 输出 affine 层.                   #
        #                                                                          #
        # IMPORTANT: For 这个 assignment, 你可以 assume 该 padding          #
        # and stride 的 first convolutional 层 are chosen so 该           #
        # **width 并 height 的 输入 are preserved**. Take a look at      #
        # start 的 损失() 函数 到 see how 该 happens.                #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate 损失 并 梯度 用于 three-层 convolutional network.

        输入 / 输出: Same API as TwoLayerNet 在 fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param 到 前向传播 用于 convolutional 层
        # Padding 并 stride chosen 到 preserve 输入 spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param 到 前向传播 用于 max-pooling 层
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO：实现 前向传播 用于 three-层 convolutional net,  #
        # 计算 类别分数 用于 X 并 storing them 在 分数          #
        # 变量.                                                                #
        #                                                                          #
        # Remember 你可以 使用 函数 defined 在 cs231n/fast_层.py 并  #
        # cs231n/层_utils.py 在 your 实现 (already imported).         #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO：实现 反向传播 用于 three-层 convolutional net, #
        # storing 损失 并 梯度 在 损失 并 grads 变量. 计算  #
        # 数据 损失 使用 softmax, 并 确保 该 grads[k] holds 梯度 #
        # for self.params[k]. 不要忘记 到 add L2 正则化!               #
        #                                                                          #
        # 注意： To ensure 该 your 实现 与参考实现匹配 并 能通过   #
        # 自动测试, 确保 该 your L2 正则化 包含 a 因子 #
        # of 0.5 到 简化表达式 用于 梯度.                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads
