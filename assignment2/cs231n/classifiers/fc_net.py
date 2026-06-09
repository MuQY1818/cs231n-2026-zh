from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    网络包含任意数量的 hidden layer、ReLU 非线性，以及 softmax loss 函数。
    本类还会把 dropout 和 batch/layer normalization 作为可选功能实现。
    对于一个 L 层网络，架构为

    {affine - [batch/层 norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中 batch/layer normalization 和 dropout 是可选的，{...} 块重复 L - 1 次。

    可学习参数存储在 self.params 字典中，并通过 Solver 类进行学习。
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
        - hidden_dims: 整数列表，给出每个 hidden layer 的大小。
        - input_dim: 整数，给出输入大小。
        - num_classes: 整数，给出要分类的类别数。
        - dropout_keep_ratio: 0 到 1 之间的标量，表示 dropout 保留比例。
            如果 dropout_keep_ratio=1，则网络完全不使用 dropout。
        - normalization: 网络使用的 normalization 类型。合法取值为
            "batchnorm"、"layernorm" 或 None，None 表示不使用 normalization。
        - reg: 标量，表示 L2 regularization 强度。
        - weight_scale: 标量，表示随机初始化权重时使用的标准差。
        - dtype: numpy datatype 对象；所有计算都会使用这个数据类型。
            float32 更快但精度较低，因此做 numerical gradient check 时应使用
            float64。
        - seed: 如果不是 None，则把这个随机种子传给 dropout layer。
            这会让 dropout layer 具有确定性，便于对模型做 gradient check。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络参数，并将所有值存入 self.params 字典。第一层的权重   #
        # 和偏置存为 W1 和 b1；第二层使用 W2 和 b2，依此类推。权重应从均值为   #
        # 0、标准差为 weight_scale 的正态分布中初始化；偏置应初始化为 0。        #
        #                                                                          #
        # 使用 batch normalization 时，第一层的缩放和平移参数存为 gamma1 和     #
        # beta1；第二层使用 gamma2 和 beta2，依此类推。缩放参数应初始化为 1，    #
        # 平移参数应初始化为 0。                                                #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 使用 dropout 时，需要向每个 dropout layer 传入 dropout_param 字典，
        # 让 layer 知道 dropout 概率和当前模式（train / test）。可以把同一个
        # dropout_param 传给每个 dropout layer。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用 batch normalization 时，需要维护 running mean 和 variance，
        # 因此要向每个 batch normalization layer 传入一个专用 bn_param 对象。
        # 第一个 batch normalization layer 的 forward pass 使用
        # self.bn_params[0]，第二个使用 self.bn_params[1]，依此类推。
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
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        返回:
        如果 y 为 None，则运行模型的 test-time forward pass 并返回:
        - scores: 形状为 (N, C) 的数组，给出分类分数，其中
            scores[i, c] 是 X[i] 属于类别 c 的分类分数。

        如果 y 不是 None，则运行 training-time forward 和 backward pass，并返回:
        - loss: 标量，表示 loss
        - grads: 字典，键与 self.params 相同，将参数名映射到 loss 关于这些
            参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 设置 batchnorm params 和 dropout param 的 train/test 模式，因为它们
        # 在训练和测试阶段的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO：实现 fully connected net 的 forward pass，计算 X 的类别分数并  #
        # 存入 scores 变量。                                                   #
        #                                                                          #
        # 使用 dropout 时，需要把 self.dropout_param 传给每个 dropout forward     #
        # pass。                                                                 #
        #                                                                          #
        # 使用 batch normalization 时，需要把 self.bn_params[0] 传给第一个        #
        # batch normalization layer 的 forward pass，把 self.bn_params[1] 传给     #
        # 第二个 batch normalization layer 的 forward pass，依此类推。             #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 如果是测试模式，则提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO：实现 fully connected net 的 backward pass。将 loss 存入 loss      #
        # 变量，并将梯度存入 grads 字典。使用 softmax 计算 data loss，并确保       #
        # grads[k] 保存 self.params[k] 的梯度。不要忘记加入 L2 regularization！   #
        #                                                                          #
        # 使用 batch/layer normalization 时，不需要对缩放和平移参数做              #
        # regularization。                                                       #
        #                                                                          #
        # 注意：为了让你的实现与参考实现匹配并通过自动测试，请确保 L2             #
        # regularization 中包含 0.5 因子，以简化梯度表达式。                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss, grads
