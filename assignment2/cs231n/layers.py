from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """计算 affine（fully connected）layer 的 forward pass。

    输入 x 的形状为 (N, d_1, ..., d_k)，包含 N 个样本组成的 minibatch。
    每个样本 x[i] 的形状为 (d_1, ..., d_k)。我们会把每个输入 reshape 成
    维度为 D = d_1 * ... * d_k 的向量，然后将其变换为维度为 M 的输出向量。

    输入:
    - x: 包含输入数据的 numpy 数组，形状为 (N, d_1, ..., d_k)
    - w: 权重 numpy 数组，形状为 (D, M)
    - b: 偏置 numpy 数组，形状为 (M,)

    返回一个 tuple:
    - out: 输出，形状为 (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO：复制你在 Assignment 1 中的解答。                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """计算 affine（fully connected）layer 的 backward pass。

    输入:
    - dout: 上游导数，形状为 (N, M)
    - cache: tuple，包含:
      - x: 输入数据，形状为 (N, d_1, ... d_k)
      - w: 权重，形状为 (D, M)
      - b: 偏置，形状为 (M,)

    返回一个 tuple:
    - dx: 关于 x 的梯度，形状为 (N, d1, ..., d_k)
    - dw: 关于 w 的梯度，形状为 (D, M)
    - db: 关于 b 的梯度，形状为 (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO：复制你在 Assignment 1 中的解答。                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """计算 rectified linear units（ReLUs）layer 的 forward pass。

    输入:
    - x: 任意形状的输入

    返回一个 tuple:
    - out: 输出，形状与 x 相同
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO：复制你在 Assignment 1 中的解答。                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """计算 rectified linear units（ReLUs）layer 的 backward pass。

    输入:
    - dout: 任意形状的上游导数
    - cache: 输入 x，形状与 dout 相同

    返回:
    - dx: 关于 x 的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # TODO：复制你在 Assignment 1 中的解答。                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """计算 softmax 分类的 loss 和梯度。

    输入:
    - x: 输入数据，形状为 (N, C)，其中 x[i, j] 是第 i 个输入属于第 j 类的分数。
    - y: 标签向量，形状为 (N,)，其中 y[i] 是 x[i] 的标签，并且
      0 <= y[i] < C

    返回一个 tuple:
    - loss: 标量 loss
    - dx: loss 关于 x 的梯度
    """
    loss, dx = None, None

    ###########################################################################
    # TODO：复制你在 Assignment 1 中的解答。                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """batch normalization 的 forward pass。

    训练时，从 minibatch 统计量计算样本均值和（未校正的）样本方差，并用它们
    对输入数据进行 normalize。训练时还会维护每个特征均值和方差的指数衰减
    running average，这些 running average 会在测试时用于 normalize 数据。

    在每个 timestep，我们基于 momentum 参数用指数衰减更新均值和方差的
    running average:

    running_均值 = momentum * running_均值 + (1 - momentum) * sample_均值
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意，batch normalization 论文建议了另一种测试时行为：使用大量训练图像
    为每个特征计算样本均值和方差，而不是使用 running average。本实现选择使用
    running average，因为它们不需要额外估计步骤；torch7 的 batch normalization
    实现也使用 running average。

    输入:
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 平移参数，形状为 (D,)
    - bn_param: 包含以下键的字典:
      - mode: 'train' 或 'test'；必需
      - eps: 用于 numerical stability 的常数
      - momentum: running mean / variance 使用的常数
      - running_mean: 形状为 (D,) 的数组，给出特征的 running mean
      - running_var: 形状为 (D,) 的数组，给出特征的 running variance

    返回一个 tuple:
    - out: 形状为 (N, D)
    - cache: backward pass 需要的值组成的 tuple
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO：实现 batch norm 的 training-time forward pass。使用 minibatch   #
        # 统计量计算均值和方差，用这些统计量 normalize 输入数据，再使用 gamma 和 #
        # beta 对 normalized data 做缩放和平移。                                #
        #                                                                     #
        # 应将输出存入变量 out。backward pass 需要的任何中间量都应存入 cache    #
        # 变量。                                                               #
        #                                                                     #
        # 还应结合 momentum 变量，使用你计算得到的样本均值和方差更新             #
        # running_mean 和 running_var。                                         #
        #                                                                     #
        # 注意：虽然需要跟踪 running variance，但 normalize 数据时应基于        #
        # standard deviation（variance 的平方根）！参考原论文                   #
        # (https://arxiv.org/abs/1502.03167) 可能会有帮助。                    #
        #######################################################################
        pass
        #######################################################################
        #                           你的代码结束                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO：实现 batch normalization 的 test-time forward pass。使用        #
        # running mean 和 variance normalize 输入数据，然后使用 gamma 和 beta   #
        # 对 normalized data 做缩放和平移。将结果存入 out 变量。                #
        #######################################################################
        pass
        #######################################################################
        #                          你的代码结束                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 将更新后的 running mean 写回 bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """batch normalization 的 backward pass。

    在这个实现中，你应该在纸上画出 batch normalization 的计算图，并通过中间节点
    反向传播梯度。

    输入:
    - dout: 上游导数，形状为 (N, D)
    - cache: 来自 batchnorm_forward 的中间量变量

    返回一个 tuple:
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 batch normalization 的 backward pass。将结果存入 dx、dgamma 和 #
    # dbeta 变量。参考原论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """batch normalization 的另一种 backward pass。

    在这个实现中，你应该在纸上推导 batch normalization backward pass 的导数，
    并尽可能化简。你应该能推导出一个用于 backward pass 的简单表达式。
    更多提示见 jupyter notebook。

    注意：这个实现应接收与 batchnorm_backward 相同的 cache 变量，但可能不会用到
    cache 中的所有值。

    输入 / 输出: 与 batchnorm_backward 相同
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 batch normalization 的 backward pass。将结果存入 dx、dgamma 和 #
    # dbeta 变量。                                                            #
    #                                                                         #
    # 计算 centered inputs 的梯度后，你应该能用一条语句计算输入的梯度；我们的  #
    # 实现可以放在一行 80 字符内。                                            #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """layer normalization 的 forward pass。

    在训练和测试时，输入数据都会按数据点进行 normalize，然后使用 gamma 和 beta
    参数进行缩放和平移；这些参数与 batch normalization 中的参数形式相同。

    注意，与 batch normalization 不同，layer normalization 在训练和测试时的行为
    完全相同，因此不需要维护任何 running average。

    输入:
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 平移参数，形状为 (D,)
    - ln_param: 包含以下键的字典:
        - eps: 用于 numerical stability 的常数

    返回一个 tuple:
    - out: 形状为 (N, D)
    - cache: backward pass 需要的值组成的 tuple
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO：实现 layer norm 的 training-time forward pass。normalize 输入数据， #
    # 然后使用 gamma 和 beta 缩放、平移 normalized data。                       #
    # 提示：可以在你的 training-time batch normalization 实现上稍作修改，并插入 #
    # 一两行位置合适的代码来完成。特别地，你能想到什么矩阵变换，让你几乎不改动 #
    # batch norm 代码就能复用它吗？                                            #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """layer normalization 的 backward pass。

    在这个实现中，可以大量复用你已经为 batch normalization 完成的工作。

    输入:
    - dout: 上游导数，形状为 (N, D)
    - cache: 来自 layernorm_forward 的中间量变量

    返回一个 tuple:
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 layer norm 的 backward pass。                                #
    #                                                                         #
    # 提示：可以在你的 training-time batch normalization 实现上稍作修改来完成。#
    # forward pass 中的提示仍然适用！                                          #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """inverted dropout 的 forward pass。

    注意，这与 vanilla dropout 版本不同。这里 p 表示保留某个 neuron 输出的概率，
    而不是丢弃某个 neuron 输出的概率。更多细节见
    http://cs231n.github.io/neural-networks-2/#reg。

    输入:
    - x: 任意形状的输入数据
    - dropout_param: 包含以下键的字典:
      - p: dropout 参数。每个 neuron 输出以概率 p 被保留。
      - mode: 'test' 或 'train'。如果 mode 为 train，则执行 dropout；
        如果 mode 为 test，则直接返回输入。
      - seed: 随机数生成器的种子。传入 seed 会让此函数具有确定性，
        这在 gradient check 时需要，但真实网络中不需要。

    Outputs:
    - out: 与 x 形状相同的数组。
    - cache: tuple (dropout_param, mask)。训练模式下，mask 是用于乘以输入的
      dropout mask；测试模式下，mask 为 None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO：实现 inverted dropout 的训练阶段 forward pass。将 dropout mask  #
        # 存入 mask 变量。                                                     #
        #######################################################################
        pass
        #######################################################################
        #                           你的代码结束                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO：实现 inverted dropout 的测试阶段 forward pass。                 #
        #######################################################################
        pass
        #######################################################################
        #                            你的代码结束                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """inverted dropout 的 backward pass。

    输入:
    - dout: 任意形状的上游导数
    - cache: 来自 dropout_forward 的 (dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO：实现 inverted dropout 的训练阶段 backward pass。                #
        #######################################################################
        pass
        #######################################################################
        #                          你的代码结束                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """convolutional layer 的 naive forward pass 实现。

    输入由 N 个数据点组成，每个数据点有 C 个通道、高度 H、宽度 W。我们用 F 个
    不同 filter 对每个输入做卷积，其中每个 filter 覆盖所有 C 个通道，高度为 HH，
    宽度为 WW。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - w: filter 权重，形状为 (F, C, HH, WW)
    - b: 偏置，形状为 (F,)
    - conv_param: 包含以下键的字典:
      - 'stride': 水平和垂直方向上相邻 receptive field 之间的像素数。
      - 'pad': 用于对输入做 zero-padding 的像素数。

    padding 时，'pad' 个 0 应沿输入的 height 和 width 轴对称放置（即两侧相等）。
    注意不要直接修改原始输入 x。

    返回一个 tuple:
    - out: 输出数据，形状为 (N, F, H', W')，其中 H' 和 W' 由下式给出
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO：实现 convolutional forward pass。                                  #
    # 提示：可以使用 np.pad 函数进行 padding。                                 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """convolutional layer 的 naive backward pass 实现。

    输入:
    - dout: 上游导数。
    - cache: tuple，包含 conv_forward_naive 中的 (x, w, b, conv_param)

    返回一个 tuple:
    - dx: 关于 x 的梯度
    - dw: 关于 w 的梯度
    - db: 关于 b 的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO：实现 convolutional backward pass。                                 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """max-pooling layer 的 naive forward pass 实现。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - pool_param: 包含以下键的字典:
      - 'pool_height': 每个 pooling region 的高度
      - 'pool_width': 每个 pooling region 的宽度
      - 'stride': 相邻 pooling region 之间的距离

    这里不需要 padding，例如可以假设:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回一个 tuple:
    - out: 输出数据，形状为 (N, C, H', W')，其中 H' 和 W' 由下式给出
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO：实现 max-pooling forward pass。                                  #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """max-pooling layer 的 naive backward pass 实现。

    输入:
    - dout: 上游导数
    - cache: tuple，包含 forward pass 中的 (x, pool_param)

    返回:
    - dx: 关于 x 的梯度
    """
    dx = None
    ###########################################################################
    # TODO：实现 max-pooling backward pass。                                 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """计算 spatial batch normalization 的 forward pass。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (C,)
    - beta: 平移参数，形状为 (C,)
    - bn_param: 包含以下键的字典:
      - mode: 'train' 或 'test'；必需
      - eps: 用于 numerical stability 的常数
      - momentum: running mean / variance 使用的常数。momentum=0 表示每个
        timestep 都完全丢弃旧信息；momentum=1 表示完全不纳入新信息。
        默认值 momentum=0.9 在大多数情况下效果不错。
      - running_mean: 形状为 (D,) 的数组，给出特征的 running mean
      - running_var: 形状为 (D,) 的数组，给出特征的 running variance

    返回一个 tuple:
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: backward pass 需要的值
    """
    out, cache = None, None

    ###########################################################################
    # TODO：实现 spatial batch normalization 的 forward pass。                 #
    #                                                                         #
    # 提示：可以通过调用上面实现的 vanilla batch normalization 来实现 spatial  #
    # batch normalization。你的实现应非常短；我们的实现少于五行。              #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """计算 spatial batch normalization 的 backward pass。

    输入:
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: 来自 forward pass 的值

    返回一个 tuple:
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: 关于缩放参数的梯度，形状为 (C,)
    - dbeta: 关于平移参数的梯度，形状为 (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO：实现 spatial batch normalization 的 backward pass。                #
    #                                                                         #
    # 提示：可以通过调用上面实现的 vanilla batch normalization 来实现 spatial  #
    # batch normalization。你的实现应非常短；我们的实现少于五行。              #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """计算 spatial group normalization 的 forward pass。
    
    与 layer normalization 不同，group normalization 会把数据中的每个条目分成
    G 个连续部分，并分别 normalize。随后对数据应用 per-feature 的平移和缩放，
    方式与 batch normalization 和 layer normalization 相同。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (1, C, 1, 1)
    - beta: 平移参数，形状为 (1, C, 1, 1)
    - G: 要拆分成的 group 数，整数，应该能整除 C
    - gn_param: 包含以下键的字典:
      - eps: 用于 numerical stability 的常数

    返回一个 tuple:
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: backward pass 需要的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO：实现 spatial group normalization 的 forward pass。                 #
    # 这会与 layer norm 的实现非常相似。特别地，思考如何变换矩阵，使主要代码   #
    # 同时类似于 train-time batch normalization 和 layer normalization！        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """计算 spatial group normalization 的 backward pass。

    输入:
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: 来自 forward pass 的值

    返回一个 tuple:
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: 关于缩放参数的梯度，形状为 (1, C, 1, 1)
    - dbeta: 关于平移参数的梯度，形状为 (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO：实现 spatial group normalization 的 backward pass。                #
    # 这会与 layer norm 的实现非常相似。                                       #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dgamma, dbeta
