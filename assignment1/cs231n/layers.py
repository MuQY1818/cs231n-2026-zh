from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    计算 前向传播 用于 an affine (fully-connected) 层.

    输入 x has 形状 (N, d_1, ..., d_k) 并 contains a minibatch 的 N
    样本, 其中 each 样本 x[i] has 形状 (d_1, ..., d_k). 我们将
    reshape each 输入 到 a vector 的 维度 D = d_1 * ... * d_k, and
    然后 transform it 到 an 输出 vector 的 维度 M.

    输入:
    - x: A numpy 数组 containing 输入 数据, 的 形状 (N, d_1, ..., d_k)
    - w: A numpy 数组 的 权重, 的 形状 (D, M)
    - b: A numpy 数组 的 偏置, 的 形状 (M,)

    返回 a tuple of:
    - out: 输出, 的 形状 (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO：实现 affine 前向传播. 将结果存储 在 out. You   #
    # 将 需要 到 reshape 输入 到 rows.                               #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    计算 反向传播 用于 an affine 层.

    输入:
    - dout: Upstream derivative, 的 形状 (N, M)
    - cache: Tuple of:
      - x: 输入 数据, 的 形状 (N, d_1, ... d_k)
      - w: Weights, 的 形状 (D, M)
      - b: Biases, 的 形状 (M,)

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 x, 的 形状 (N, d1, ..., d_k)
    - dw: Gradient 使用 respect 到 w, 的 形状 (D, M)
    - db: Gradient 使用 respect 到 b, 的 形状 (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO：实现 affine 反向传播.                               #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    计算 前向传播 用于 a 层 的 rectified linear units (ReLUs).

    输入:
    - x: 输入, 的 any 形状

    返回 a tuple of:
    - out: Output, 的 same 形状 as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO：实现 ReLU 前向传播.                                  #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    计算 反向传播 用于 a 层 的 rectified linear units (ReLUs).

    输入:
    - dout: Upstream derivatives, 的 any 形状
    - cache: 输入 x, 的 same 形状 as dout

    返回:
    - dx: Gradient 使用 respect 到 x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO：实现 ReLU 反向传播.                                 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    前向传播 用于 batch 归一化.

    训练时 样本均值 并 (uncorrected) 样本方差 are
    计算得到的 来自 minibatch 统计量 并 使用 到 normalize 输入的 数据.
    训练时 we also keep an exponenti所有y decaying running 均值 的 the
    均值 并 方差 的 each 特征, 并 这些 averages are 使用 到 normalize
    数据 at 测试时.

    At each timestep we update running averages 用于 均值 并 方差 使用
    an exponential decay 基于 momentum parameter:

    running_均值 = momentum * running_均值 + (1 - momentum) * sample_均值
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note 该 batch 归一化 paper suggests a different 测试时
    behavior: they 计算 样本均值 并 方差 用于 each 特征 使用 a
    large 数量 训练 images rather than 使用 a running average. For
    这个 实现 我们已经 chosen 到 使用 running averages instead since
    they 不要 require an additional estimation step; torch7
    实现 的 batch 归一化 also 使用s running averages.

    输入:
    - x: Data 的 形状 (N, D)
    - gamma: Scale parameter 的 形状 (D,)
    - beta: Shift paremeter 的 形状 (D,)
    - bn_param: Dictionary 使用 following keys:
      - 模式: '训练' or '测试'; required
      - eps: Constant 用于 numeric stability
      - momentum: Constant 用于 running 均值 / 方差.
      - running_均值: Array 的 形状 (D,) giving running 均值 的 特征
      - running_var Array 的 形状 (D,) giving running 方差 的 特征

    返回 a tuple of:
    - out: 的 形状 (N, D)
    - cache: 一个 tuple，包含 值 需要 在 反向传播
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
        # TODO：实现 训练时 前向传播 用于 batch norm.      #
        # 使用 minibatch 统计量 到 计算 均值 并 方差, 使用      #
        # 这些 统计量 到 normalize 输入的 数据, 并 缩放 并      #
        # 平移 归一化后的 数据 使用 gamma 并 beta.                     #
        #                                                                     #
        # 你应该 存储 输出 在 变量 out. Any 中间量  #
        # 该 you 需要 用于 反向传播 应为 存储 在 cache   #
        # 变量.                                                           #
        #                                                                     #
        # 你应该 also 使用 your 计算得到的 样本均值 并 方差 together #
        # 使用 momentum 变量 到 update running 均值 并 running   #
        # 方差, storing your 结果 在 running_均值 并 running_var   #
        # 变量.                                                          #
        #                                                                     #
        # Note 该 though you 应为 keeping track 的 running         #
        # 方差, 你应该 normalize 数据 基于 standard       #
        # deviation (square root 的 方差) instead!                        #
        # Referencing 原始论文 (https://arxiv.org/abs/1502.03167)   #
        # 可能 可能会有帮助.                                          #
        #######################################################################
        pass
        #######################################################################
        #                           你的代码结束                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO：实现 测试时 前向传播 用于 batch 归一化. #
        # 使用 running 均值 并 方差 到 normalize 输入的 数据,   #
        # 然后 缩放 并 平移 归一化后的 数据 使用 gamma 并 beta.      #
        # 将结果存储 在 out 变量.                               #
        #######################################################################
        pass
        #######################################################################
        #                          你的代码结束                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 将更新后的 running 均值 写回 bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    反向传播 用于 batch 归一化.

    For 这个 实现, 你应该 write out a computation graph for
    batch 归一化 on paper 并 propagate 梯度 反向 through
    intermediate nodes.

    输入:
    - dout: Upstream derivatives, 的 形状 (N, D)
    - cache: Variable 的 中间量 来自 batchnorm_前向.

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 输入 x, 的 形状 (N, D)
    - dgamma: Gradient 使用 respect 到 缩放 parameter gamma, 的 形状 (D,)
    - dbeta: Gradient 使用 respect 到 平移 parameter beta, 的 形状 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 反向传播 用于 batch 归一化. 存储    #
    # 结果 在 dx, dgamma, 并 dbeta 变量.                         #
    # Referencing 原始论文 (https://arxiv.org/abs/1502.03167)       #
    # 可能 可能会有帮助.                                              #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative 反向传播 用于 batch 归一化.

    For 这个 实现 你应该 work out derivatives 用于 batch
    normalizaton 反向传播 on paper 并 简化 as much as possible. You
    应为 able 到 derive a simple 表达式 用于 反向传播.
    See jupyter notebook 用于 more hints.

    注意： This 实现 应该 expect 到 receive same cache 变量
    as batchnorm_反向, but 可能 不要使用 所有 的 值 在 cache.

    输入 / 输出: Same as batchnorm_反向
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 反向传播 用于 batch 归一化. 存储    #
    # 结果 在 dx, dgamma, 并 dbeta 变量.                         #
    #                                                                         #
    # After 计算 梯度 使用 respect 到 centered 输入, you   #
    # 应为 able 到 计算 梯度 使用 respect 到 输入 在 a     #
    # single statement; our 实现 fits on a single 80-character line.#
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    前向传播 用于 层 归一化.

    During both 训练 并 测试时, 输入的 数据 is 归一化后的 per 数据-点,
    before being 缩放 by gamma 并 beta 参数 identical 到 该 的 batch 归一化.

    Note 该 在 contrast 到 batch 归一化, behavior during 训练 并 测试时 for
    层 归一化 are identical, 并 we 不要 需要 到 keep track 的 running averages
    of any sort.

    输入:
    - x: Data 的 形状 (N, D)
    - gamma: Scale parameter 的 形状 (D,)
    - beta: Shift paremeter 的 形状 (D,)
    - ln_param: Dictionary 使用 following keys:
        - eps: Constant 用于 numeric stability

    返回 a tuple of:
    - out: 的 形状 (N, D)
    - cache: 一个 tuple，包含 值 需要 在 反向传播
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO：实现 训练时 前向传播 用于 层 norm.          #
    # Normalize 输入的 数据, 并 缩放 并  平移 归一化后的 数据   #
    #  使用 gamma 并 beta.                                                  #
    # 提示： 这个 可以 be done by slightly modifying your 训练时         #
    # 实现 的  batch 归一化, 并 inserting a line or two 的  #
    # well-placed code. In particular, 可以 you think 的 any 矩阵            #
    # transformations you could perform, 该 would enable you 到 copy 在   #
    # batch norm code 并 leave it almost unchanged?                      #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    反向传播 用于 层 归一化.

    For 这个 实现, 你可以 heavily rely on work you've done already
    for batch 归一化.

    输入:
    - dout: Upstream derivatives, 的 形状 (N, D)
    - cache: Variable 的 中间量 来自 层norm_前向.

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 输入 x, 的 形状 (N, D)
    - dgamma: Gradient 使用 respect 到 缩放 parameter gamma, 的 形状 (D,)
    - dbeta: Gradient 使用 respect 到 平移 parameter beta, 的 形状 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO：实现 反向传播 用于 层 norm.                       #
    #                                                                         #
    # 提示： 这个 可以 be done by slightly modifying your 训练时         #
    # 实现 的 batch 归一化. hints 到 前向传播    #
    # still apply!                                                            #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs 前向传播 用于 (inverted) dropout.

    输入:
    - x: 输入 数据, 的 any 形状
    - dropout_param: A 字典 使用 following keys:
      - p: Dropout parameter. We keep each neuron 输出 使用 probability p.
      - 模式: '测试' or '训练'. If 模式 is 训练, 然后 perform dropout;
        if 模式 is 测试, 然后 just return 输入.
      - seed: Seed 用于 random number generator. Passing seed makes 这个
        函数 deterministic, which is 需要 用于 梯度 checking but not
        in real networks.

    Outputs:
    - out: Array 的 same 形状 as x.
    - cache: tuple (dropout_param, mask). In 训练 模式, mask is dropout
      mask 该 was 使用 到 multiply 输入; 在 测试 模式, mask is None.

    注意： Please implement **inverted** dropout, not vanilla version 的 dropout.
    See http://cs231n.github.io/neural-networks-2/#reg 用于 more details.

    NOTE 2: Keep 在 mind 该 p is probability 的 **keep** a neuron
    输出; 这个 可能 be contrary 到 some sources, 其中 it is referred to
    as probability 的 dropping a neuron 输出.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO：实现 训练 phase 前向传播 用于 inverted dropout.   #
        # 将 dropout mask 存储 在 mask 变量.                        #
        #######################################################################
        pass
        #######################################################################
        #                           你的代码结束                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO：实现 测试 phase 前向传播 用于 inverted dropout.   #
        #######################################################################
        pass
        #######################################################################
        #                            你的代码结束                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform 反向传播 用于 (inverted) dropout.

    输入:
    - dout: Upstream derivatives, 的 any 形状
    - cache: (dropout_param, mask) 来自 dropout_前向.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO：实现 训练 phase 反向传播 用于 inverted dropout   #
        #######################################################################
        pass
        #######################################################################
        #                          你的代码结束                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive 实现 的 前向传播 用于 a convolutional 层.

    输入 consists 的 N 数据 点, each 使用 C channels, height H and
    width W. We convolve each 输入 使用 F different filters, 其中 each filter
    spans 所有 C channels 并 has height HH 并 width WW.

    输入:
    - x: 输入 数据 的 形状 (N, C, H, W)
    - w: Filter 权重 的 形状 (F, C, HH, WW)
    - b: Biases, 的 形状 (F,)
    - conv_param: A 字典 使用 following keys:
      - 'stride': 数量 pixels between adjacent receptive fields 在 the
        horizontal 并 vertical directions.
      - 'pad': 数量 pixels 该 将 be 使用 到 zero-pad 输入.


    During padding, 'pad' zeros 应为 placed symmetri调用y (i.e equ所有y on both sides)
    along height 并 width axes 的 输入. Be careful not 到 modfiy original
    输入 x directly.

    返回 a tuple of:
    - out: Output 数据, 的 形状 (N, F, H', W') 其中 H' 并 W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO：实现 convolutional 前向传播.                         #
    # 提示： 你可以 使用 函数 np.pad 用于 padding.                      #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive 实现 的 反向传播 用于 a convolutional 层.

    输入:
    - dout: Upstream derivatives.
    - cache: 一个 tuple，包含 (x, w, b, conv_param) as 在 conv_前向_naive

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 x
    - dw: Gradient 使用 respect 到 w
    - db: Gradient 使用 respect 到 b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO：实现 convolutional 反向传播.                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive 实现 的 前向传播 用于 a max-pooling 层.

    输入:
    - x: 输入 数据, 的 形状 (N, C, H, W)
    - pool_param: 字典 使用 following keys:
      - 'pool_height': height 的 each pooling region
      - 'pool_width': width 的 each pooling region
      - 'stride': 距离 between adjacent pooling regions

    No padding is necessary here, eg 你可以 assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回 a tuple of:
    - out: Output 数据, 的 形状 (N, C, H', W') 其中 H' 并 W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO：实现 max-pooling 前向传播                            #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive 实现 的 反向传播 用于 a max-pooling 层.

    输入:
    - dout: Upstream derivatives
    - cache: 一个 tuple，包含 (x, pool_param) as 在 前向传播.

    返回:
    - dx: Gradient 使用 respect 到 x
    """
    dx = None
    ###########################################################################
    # TODO：实现 max-pooling 反向传播                           #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    计算 前向传播 用于 spatial batch 归一化.

    输入:
    - x: 输入 数据 的 形状 (N, C, H, W)
    - gamma: Scale parameter, 的 形状 (C,)
    - beta: Shift parameter, 的 形状 (C,)
    - bn_param: Dictionary 使用 following keys:
      - 模式: '训练' or '测试'; required
      - eps: Constant 用于 numeric stability
      - momentum: Constant 用于 running 均值 / 方差. momentum=0 均值 该
        old information is discarded completely at every time step, while
        momentum=1 均值 该 new information is never incorporated. The
        default 的 momentum=0.9 应该 work well 在 most situations.
      - running_均值: Array 的 形状 (D,) giving running 均值 的 特征
      - running_var Array 的 形状 (D,) giving running 方差 的 特征

    返回 a tuple of:
    - out: Output 数据, 的 形状 (N, C, H, W)
    - cache: Values 需要 用于 反向传播
    """
    out, cache = None, None

    ###########################################################################
    # TODO：实现 前向传播 用于 spatial batch 归一化.       #
    #                                                                         #
    # 提示： 你可以 implement spatial batch 归一化 by 调用      #
    # vanilla version 的 batch 归一化 you implemented above.           #
    # Your 实现 应为 very short; ours is 小于 five lines. #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    计算 反向传播 用于 spatial batch 归一化.

    输入:
    - dout: Upstream derivatives, 的 形状 (N, C, H, W)
    - cache: Values 来自 前向传播

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 输入, 的 形状 (N, C, H, W)
    - dgamma: Gradient 使用 respect 到 缩放 parameter, 的 形状 (C,)
    - dbeta: Gradient 使用 respect 到 平移 parameter, 的 形状 (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO：实现 反向传播 用于 spatial batch 归一化.      #
    #                                                                         #
    # 提示： 你可以 implement spatial batch 归一化 by 调用      #
    # vanilla version 的 batch 归一化 you implemented above.           #
    # Your 实现 应为 very short; ours is 小于 five lines. #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    计算 前向传播 用于 spatial group 归一化.
    In contrast 到 层 归一化, group 归一化 splits each entry
    in 数据 到 G contiguous pieces, which it 然后 normalizes independently.
    Per 特征 平移 并 scaling are 然后 applied 到 数据, 在 a manner identical 到 该 的 batch 归一化 并 层 归一化.

    输入:
    - x: 输入 数据 的 形状 (N, C, H, W)
    - gamma: Scale parameter, 的 形状 (1, C, 1, 1)
    - beta: Shift parameter, 的 形状 (1, C, 1, 1)
    - G: Integer mumber 的 groups 到 split 到, 应为 a divisor 的 C
    - gn_param: Dictionary 使用 following keys:
      - eps: Constant 用于 numeric stability

    返回 a tuple of:
    - out: Output 数据, 的 形状 (N, C, H, W)
    - cache: Values 需要 用于 反向传播
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO：实现 前向传播 用于 spatial group 归一化.       #
    # This 将 be extremely similar 到 层 norm 实现.        #
    # In particular, think about how you could transform 矩阵 so 该   #
    # bulk 的 code is similar 到 both 训练-time batch 归一化  #
    # and 层 归一化!                                                #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    计算 反向传播 用于 spatial group 归一化.

    输入:
    - dout: Upstream derivatives, 的 形状 (N, C, H, W)
    - cache: Values 来自 前向传播

    返回 a tuple of:
    - dx: Gradient 使用 respect 到 输入, 的 形状 (N, C, H, W)
    - dgamma: Gradient 使用 respect 到 缩放 parameter, 的 形状 (1, C, 1, 1)
    - dbeta: Gradient 使用 respect 到 平移 parameter, 的 形状 (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO：实现 反向传播 用于 spatial group 归一化.      #
    # This 将 be extremely similar 到 层 norm 实现.        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    计算 损失 并 梯度 使用 用于 multi类别 SVM 分类.

    输入:
    - x: 输入 数据, 的 形状 (N, C) 其中 x[i, j] is score 用于 jth
      类别 用于 ith 输入.
    - y: Vector 的 标签, 的 形状 (N,) 其中 y[i] is 标签 用于 x[i] and
      0 <= y[i] < C

    返回 a tuple of:
    - 损失: Scalar giving 损失
    - dx: Gradient 的 损失 使用 respect 到 x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO：复制你在 A1 中的解答。
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    计算 损失 并 梯度 用于 softmax 分类.

    输入:
    - x: 输入 数据, 的 形状 (N, C) 其中 x[i, j] is score 用于 jth
      类别 用于 ith 输入.
    - y: Vector 的 标签, 的 形状 (N,) 其中 y[i] is 标签 用于 x[i] and
      0 <= y[i] < C

    返回 a tuple of:
    - 损失: Scalar giving 损失
    - dx: Gradient 的 损失 使用 respect 到 x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO：复制你在 A1 中的解答。
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    return loss, dx
