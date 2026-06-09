"""本文件定义循环神经网络中常用的层类型。"""
import torch


def affine_forward(x, w, b):
    """计算 前向传播 用于 an affine (fully connected) 层.

    输入 x has 形状 (N, d_1, ..., d_k) 并 contains a minibatch 的 N
    样本, 其中 each 样本 x[i] has 形状 (d_1, ..., d_k). 我们将
    reshape each 输入 到 a vector 的 维度 D = d_1 * ... * d_k, and
    然后 transform it 到 an 输出 vector 的 维度 M.

    输入:
    - x: A torch 数组 containing 输入 数据, 的 形状 (N, d_1, ..., d_k)
    - w: A torch 数组 的 权重, 的 形状 (D, M)
    - b: A torch 数组 的 偏置, 的 形状 (M,)

    返回 a tuple of:
    - out: 输出, 的 形状 (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run 前向传播 用于 a single timestep 的 a vanilla RNN 使用 a tanh activation 函数.

    输入 数据 has 维度 D, hidden state has 维度 H,
    and minibatch is 的 size N.

    输入:
    - x: 输入 数据 用于 这个 timestep, 的 形状 (N, D)
    - prev_h: Hidden state 来自 previous timestep, 的 形状 (N, H)
    - Wx: Weight 矩阵 用于 输入-to-hidden connections, 的 形状 (D, H)
    - Wh: Weight 矩阵 用于 hidden-to-hidden connections, 的 形状 (H, H)
    - b: Biases 的 形状 (H,)

    返回 a tuple of:
    - next_h: Next hidden state, 的 形状 (N, H)
    """
    next_h = None
    ##############################################################################
    # TODO：实现 a single 前向 step 用于 vanilla RNN.                 #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN 前向 on an entire sequence 的 数据.
    
    We assume an 输入 sequence composed 的 T vectors, each 的 维度 D. RNN 使用s a hidden
    size 的 H, 并 we work 在 a minibatch containing N sequences. After running RNN 前向,
    we return hidden states 用于 所有 timesteps.

    输入:
    - x: 输入 数据 用于 entire timeseries, 的 形状 (N, T, D)
    - h0: Initial hidden state, 的 形状 (N, H)
    - Wx: Weight 矩阵 用于 输入-to-hidden connections, 的 形状 (D, H)
    - Wh: Weight 矩阵 用于 hidden-to-hidden connections, 的 形状 (H, H)
    - b: Biases 的 形状 (H,)

    返回 a tuple of:
    - h: Hidden states 用于 entire timeseries, 的 形状 (N, T, H)
    """
    h = None
    ##############################################################################
    # TODO：实现 前向传播 用于 a vanilla RNN running on a sequence 的    #
    # 输入 数据. 你应该 使用 rnn_step_前向 函数 该 you defined  #
    # above. 你可以 使用 a 用于 loop 到 help 计算 前向传播.            #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return h


def word_embedding_forward(x, W):
    """前向传播 用于 word embeddings.
    
    We operate on minibatches 的 size N 其中
    each sequence has length T. We assume a vocabulary 的 V words, assigning each
    word 到 a vector 的 维度 D.

    输入:
    - x: Integer 数组 的 形状 (N, T) giving indices 的 words. Each element idx
      of x muxt be 在 range 0 <= idx < V.
    - W: Weight 矩阵 的 形状 (V, D) giving word vectors 用于 所有 words.

    返回 a tuple of:
    - out: Array 的 形状 (N, T, D) giving word vectors 用于 所有 输入 words.
    """
    out = None
    ##############################################################################
    # TODO：实现 前向传播 用于 word embeddings.                      #
    #                                                                            #
    # 提示： This 可以 be done 在 one line 使用 Pytorch's 数组 indexing.         #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """前向传播 用于 a single timestep 的 an LSTM.

    输入 数据 has 维度 D, hidden state has 维度 H, 并 我们使用
    a minibatch size 的 N.

    Note 该 a sigmoid() 函数 has already been provided 用于 you 在 这个 file.

    输入:
    - x: 输入 数据, 的 形状 (N, D)
    - prev_h: Previous hidden state, 的 形状 (N, H)
    - prev_c: previous cell state, 的 形状 (N, H)
    - Wx: 输入-to-hidden 权重, 的 形状 (D, 4H)
    - Wh: Hidden-to-hidden 权重, 的 形状 (H, 4H)
    - b: Biases, 的 形状 (4H,)

    返回 a tuple of:
    - next_h: Next hidden state, 的 形状 (N, H)
    - next_c: Next cell state, 的 形状 (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO：实现 前向传播 用于 a single timestep 的 an LSTM.        #
    # 你可以 want 到 使用 numeri调用y stable sigmoid 实现 above.  #
    #############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """前向传播 用于 an LSTM 在 an entire sequence 的 数据.
    
    We assume an 输入 sequence composed 的 T vectors, each 的 维度 D. LSTM 使用s a hidden
    size 的 H, 并 we work 在 a minibatch containing N sequences. After running LSTM 前向,
    we return hidden states 用于 所有 timesteps.

    Note 该 initial cell state is passed as 输入, but initial cell state is set 到 zero.
    Also note 该 cell state is not returned; it is an internal 变量 到 LSTM 并 is not
    accessed 来自 outside.

    输入:
    - x: 输入 数据 的 形状 (N, T, D)
    - h0: Initial hidden state 的 形状 (N, H)
    - Wx: Weights 用于 输入-to-hidden connections, 的 形状 (D, 4H)
    - Wh: Weights 用于 hidden-to-hidden connections, 的 形状 (H, 4H)
    - b: Biases 的 形状 (4H,)

    返回 a tuple of:
    - h: Hidden states 用于 所有 timesteps 的 所有 sequences, 的 形状 (N, T, H)
    """
    h = None
    #############################################################################
    # TODO：实现 前向传播 用于 an LSTM 在 an entire timeseries.   #
    # 你应该 使用 lstm_step_前向 函数 该 you just defined.      #
    #############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################

    return h


def temporal_affine_forward(x, w, b):
    """前向传播 用于 a temporal affine 层.
    
    输入 is a set 的 D-维度al
    vectors arranged 到 a minibatch 的 N timeseries, each 的 length T. 我们使用
    an affine 函数 到 transform each 的 those vectors 到 a new vector of
    维度 M.

    输入:
    - x: 输入 数据 的 形状 (N, T, D)
    - w: Weights 的 形状 (D, M)
    - b: Biases 的 形状 (M,)

    返回 a tuple of:
    - out: Output 数据 的 形状 (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version 的 softmax 损失 用于 使用 在 RNNs.
    
    We assume 该 we are making 预测 在 a vocabulary 的 size V 用于 each timestep 的 a
    timeseries 的 length T, 在 a minibatch 的 size N. 输入 x gives 分数 用于 所有 vocabulary
    elements at 所有 timesteps, 并 y gives indices 的 ground-truth element at each timestep.
    我们使用 a cross-entropy 损失 at each timestep, summing 损失 在 所有 timesteps 并 averaging
    across minibatch.

    As an additional complication, we may want 到 ignore 模型 输出 at some timesteps, since
    sequences 的 different length may have been combined 到 a minibatch 并 padded 使用 NULL
    tokens. optional mask argument tells us which elements 应该 contribute 到 损失.

    输入:
    - x: 输入 分数, 的 形状 (N, T, V)
    - y: Ground-truth indices, 的 形状 (N, T) 其中 each element is 在 range
         0 <= y[i, t] < V
    - mask: Boolean 数组 的 形状 (N, T) 其中 mask[i, t] tells whether or not
      分数 at x[i, t] 应该 contribute 到 损失.

    返回 a tuple of:
    - 损失: Scalar giving 损失
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
