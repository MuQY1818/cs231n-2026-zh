from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    三层 convolutional network，使用如下架构:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    该网络处理形状为 (N, C, H, W) 的数据 minibatch，其中包含 N 张图像，
    每张图像有 C 个输入通道、高度 H、宽度 W。
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
        初始化一个新网络。

        输入:
        - input_dim: 元组 (C, H, W)，给出输入数据大小。
        - num_filters: convolutional layer 中使用的 filter 数量。
        - filter_size: convolutional layer 中 filter 的宽度和高度。
        - hidden_dim: fully-connected hidden layer 中的单元数量。
        - num_classes: 最后一个 affine layer 输出的分数数量。
        - weight_scale: 标量，表示随机初始化权重时使用的标准差。
        - reg: 标量，表示 L2 regularization 强度。
        - dtype: 计算时使用的 numpy datatype。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: 初始化三层 convolutional network 的权重和偏置。权重应从均值为   #
        # 0.0、标准差为 weight_scale 的 Gaussian 分布中初始化；偏置应初始化为    #
        # 0。所有权重和偏置都应存入 self.params 字典。convolutional layer 的     #
        # 权重和偏置使用键 'W1' 和 'b1'；hidden affine layer 使用 'W2' 和        #
        # 'b2'；输出 affine layer 使用 'W3' 和 'b3'。                            #
        #                                                                          #
        # 重要：在本作业中，可以假设第一个 convolutional layer 的 padding 和     #
        # stride 会被设置为**保持输入的宽度和高度不变**。查看 loss() 函数开头，  #
        # 可以看到这是如何实现的。                                                #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        计算三层 convolutional network 的 loss 和梯度。

        输入 / 输出: API 与 fc_net.py 中的 TwoLayerNet 相同。
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # 将 conv_param 传给 convolutional layer 的 forward pass。
        # padding 和 stride 被设置为保持输入的空间尺寸不变。
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # 将 pool_param 传给 max-pooling layer 的 forward pass。
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO：实现 three-layer convolutional net 的 forward pass，计算 X 的    #
        # 类别分数并存入 scores 变量。                                           #
        #                                                                          #
        # 记住，你可以在实现中使用 cs231n/fast_layers.py 和                      #
        # cs231n/layer_utils.py 中定义的函数（已导入）。                         #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO：实现 three-layer convolutional net 的 backward pass，将 loss 和  #
        # 梯度分别存入 loss 和 grads 变量。使用 softmax 计算 data loss，并确保   #
        # grads[k] 保存 self.params[k] 的梯度。不要忘记加入 L2 regularization！  #
        #                                                                          #
        # 注意：为了让你的实现与参考实现匹配并通过自动测试，请确保 L2             #
        # regularization 中包含 0.5 因子，以简化梯度表达式。                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads
