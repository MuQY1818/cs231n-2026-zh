"""本文件定义循环神经网络中常用的层类型。"""
import torch


def affine_forward(x, w, b):
    """计算 affine（fully connected）layer 的 forward pass。

    输入 x 的形状为 (N, d_1, ..., d_k)，包含 N 个样本组成的 minibatch。
    每个样本 x[i] 的形状为 (d_1, ..., d_k)。我们会把每个输入 reshape 成
    维度为 D = d_1 * ... * d_k 的向量，然后将其变换为维度为 M 的输出向量。

    输入:
    - x: 包含输入数据的 torch 数组，形状为 (N, d_1, ..., d_k)
    - w: 权重 torch 数组，形状为 (D, M)
    - b: 偏置 torch 数组，形状为 (M,)

    返回:
    - out: 输出，形状为 (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """运行 vanilla RNN 单个 timestep 的 forward pass，使用 tanh activation 函数。

    输入数据维度为 D，hidden state 维度为 H，minibatch 大小为 N。

    输入:
    - x: 当前 timestep 的输入数据，形状为 (N, D)
    - prev_h: 来自 previous timestep 的 hidden state，形状为 (N, H)
    - Wx: input-to-hidden connection 的权重矩阵，形状为 (D, H)
    - Wh: hidden-to-hidden connection 的权重矩阵，形状为 (H, H)
    - b: 偏置，形状为 (H,)

    返回:
    - next_h: next hidden state，形状为 (N, H)
    """
    next_h = None
    ##############################################################################
    # TODO：实现 vanilla RNN 的单个 forward step。                              #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """在整个数据序列上运行 vanilla RNN forward pass。
    
    假设输入序列由 T 个向量组成，每个向量维度为 D。RNN 使用大小为 H 的 hidden
    state，并在包含 N 个序列的 minibatch 上工作。运行 RNN forward pass 后，返回
    所有 timestep 的 hidden states。

    输入:
    - x: 整个 timeseries 的输入数据，形状为 (N, T, D)
    - h0: initial hidden state，形状为 (N, H)
    - Wx: input-to-hidden connection 的权重矩阵，形状为 (D, H)
    - Wh: hidden-to-hidden connection 的权重矩阵，形状为 (H, H)
    - b: 偏置，形状为 (H,)

    返回:
    - h: 整个 timeseries 的 hidden states，形状为 (N, T, H)
    """
    h = None
    ##############################################################################
    # TODO：实现在输入数据序列上运行的 vanilla RNN forward pass。应使用上面定义的 #
    # rnn_step_forward 函数。可以使用 for 循环辅助计算 forward pass。             #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return h


def word_embedding_forward(x, W):
    """word embeddings 的 forward pass。
    
    在大小为 N 的 minibatch 上操作，其中每个序列长度为 T。假设词表有 V 个词，
    每个词分配一个维度为 D 的向量。

    输入:
    - x: 整数数组，形状为 (N, T)，给出 word index。x 的每个元素 idx 必须满足
      0 <= idx < V。
    - W: 权重矩阵，形状为 (V, D)，给出所有词的 word vector。

    返回:
    - out: 形状为 (N, T, D) 的数组，给出所有输入词的 word vector。
    """
    out = None
    ##############################################################################
    # TODO：实现 word embeddings 的 forward pass。                            #
    #                                                                            #
    # 提示：这可以用一行 PyTorch array indexing 完成。                         #
    ##############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """LSTM 单个 timestep 的 forward pass。

    输入数据维度为 D，hidden state 维度为 H，minibatch 大小为 N。

    注意，本文件已经提供 sigmoid() 函数。

    输入:
    - x: 输入数据，形状为 (N, D)
    - prev_h: previous hidden state，形状为 (N, H)
    - prev_c: previous cell state，形状为 (N, H)
    - Wx: input-to-hidden 权重，形状为 (D, 4H)
    - Wh: hidden-to-hidden 权重，形状为 (H, 4H)
    - b: 偏置，形状为 (4H,)

    返回:
    - next_h: next hidden state，形状为 (N, H)
    - next_c: next cell state，形状为 (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO：实现 LSTM 单个 timestep 的 forward pass。                         #
    # 可以使用上面提供的 numerically stable sigmoid 实现。                    #
    #############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """在整个数据序列上运行 LSTM forward pass。
    
    假设输入序列由 T 个向量组成，每个向量维度为 D。LSTM 使用大小为 H 的 hidden
    state，并在包含 N 个序列的 minibatch 上工作。运行 LSTM forward pass 后，返回
    所有 timestep 的 hidden states。

    注意，initial cell state 不作为输入传入，而是设置为 0。还要注意，cell state
    不会返回；它是 LSTM 的内部变量，外部无法访问。

    输入:
    - x: 输入数据，形状为 (N, T, D)
    - h0: initial hidden state，形状为 (N, H)
    - Wx: input-to-hidden connection 的权重，形状为 (D, 4H)
    - Wh: hidden-to-hidden connection 的权重，形状为 (H, 4H)
    - b: 偏置，形状为 (4H,)

    返回:
    - h: 所有序列在所有 timestep 的 hidden states，形状为 (N, T, H)
    """
    h = None
    #############################################################################
    # TODO：实现整个 timeseries 上的 LSTM forward pass。应使用刚定义的          #
    # lstm_step_forward 函数。                                                 #
    #############################################################################

    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################

    return h


def temporal_affine_forward(x, w, b):
    """temporal affine layer 的 forward pass。
    
    输入是一组 D 维向量，被组织成包含 N 个 timeseries 的 minibatch，每个
    timeseries 长度为 T。我们使用 affine 函数把这些向量分别变换为维度为 M 的
    新向量。

    输入:
    - x: 输入数据，形状为 (N, T, D)
    - w: 权重，形状为 (D, M)
    - b: 偏置，形状为 (M,)

    返回:
    - out: 输出数据，形状为 (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """用于 RNN 的 temporal softmax loss。
    
    假设我们在大小为 N 的 minibatch 上，对长度为 T 的 timeseries 的每个 timestep
    都在大小为 V 的词表上做预测。输入 x 给出所有 timestep 上所有词表元素的分数，
    y 给出每个 timestep 的 ground-truth 元素索引。我们在每个 timestep 使用
    cross-entropy loss，将所有 timestep 的 loss 求和，并在 minibatch 上取平均。

    额外的复杂点是，我们可能希望忽略某些 timestep 上的模型输出，因为不同长度的
    序列可能被合并到一个 minibatch 中，并用 NULL token 做 padding。可选的 mask
    参数会告诉我们哪些元素应计入 loss。

    输入:
    - x: 输入分数，形状为 (N, T, V)
    - y: ground-truth indices，形状为 (N, T)，其中每个元素都在范围
         0 <= y[i, t] < V
    - mask: boolean 数组，形状为 (N, T)，其中 mask[i, t] 表示 x[i, t] 处的
      分数是否应计入 loss。

    返回:
    - loss: 标量 loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
