import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    CaptioningRNN 使用 recurrent neural network 根据图像特征生成 captions。

    RNN 接收大小为 D 的输入向量，词表大小为 V，处理长度为 T 的序列；
    RNN hidden dimension 为 H，word vector dimension 为 W，并在大小为 N 的
    minibatch 上运行。

    注意，CaptioningRNN 不使用任何 regularization。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        构造一个新的 CaptioningRNN 实例。

        输入:
        - word_to_idx: 表示词表的字典，包含 V 个条目，将每个字符串映射到
          [0, V) 范围内的唯一整数。
        - input_dim: 输入图像特征向量的维度 D。
        - wordvec_dim: word vector 的维度 W。
        - hidden_dim: RNN hidden state 的维度 H。
        - cell_type: 使用的 RNN 类型，可以是 'rnn' 或 'lstm'。
        - dtype: 使用的 numpy datatype；训练时使用 float32，numerical
          gradient check 时使用 float64。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # 初始化词向量
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # 初始化 CNN 到 hidden state 的投影参数
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # 初始化 RNN 参数
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # 初始化 hidden-to-vocab 输出权重
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # 将参数转换为正确 dtype
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        计算 RNN 的 training-time loss。输入图像特征及其 ground-truth
        captions，使用 RNN（或 LSTM）计算所有参数上的 loss。

        输入:
        - features: 输入图像特征，形状为 (N, D)
        - captions: ground-truth captions；整数数组，形状为 (N, T + 1)，
          其中每个元素都在 0 <= y[i, t] < V 范围内

        返回:
        - loss: 标量 loss
        """
        # 将 captions 切成两段：captions_in 包含除最后一个词外的所有词，会作为
        # RNN 的输入；captions_out 包含除第一个词外的所有词，是期望 RNN 生成的
        # 输出。两者相差一个时间步，因为 RNN 在接收词 t 后应生成词 (t+1)。
        # captions_in 的第一个元素是 START token，captions_out 的第一个元素是
        # 第一个真实词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 你会用到这个 mask
        mask = captions_out != self._null

        # 从图像特征到 initial hidden state 的 affine transform 的权重和偏置
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding 矩阵
        W_embed = self.params["W_embed"]

        # RNN 的 input-to-hidden、hidden-to-hidden 权重和偏置
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # hidden-to-vocab transformation 的权重和偏置
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO：实现 CaptioningRNN 的 forward pass。                           #
        # 在 forward pass 中需要完成以下步骤：                                  #
        # (1) 对图像特征使用 affine transformation，计算 initial hidden state。  #
        #     这应产生一个形状为 (N, H) 的数组。                                #
        # (2) 使用 word embedding layer 将 captions_in 中的 word index 转换为    #
        #     vector，得到形状为 (N, T, W) 的数组。                              #
        # (3) 根据 self.cell_type 使用 vanilla RNN 或 LSTM 处理输入 word vector  #
        #     序列，并为所有 timestep 生成 hidden state vector，得到形状为       #
        #     (N, T, H) 的数组。                                                #
        # (4) 使用 (temporal) affine transformation 基于 hidden states 计算每个   #
        #     timestep 上词表中所有词的分数，得到形状为 (N, T, V) 的数组。       #
        # (5) 使用 (temporal) softmax 根据 captions_out 计算 loss，并通过上面的   #
        #     mask 忽略输出词为 <NULL> 的位置。                                  #
        #                                                                          #       
        # 请确保你的实现不依赖输入 tensor 的具体数据类型。                       #
        #                                                                          #
        # 不需要考虑对权重或其梯度做 regularization！                            #
        #                                                                          #
        # 也不需要实现 backward pass。                                           #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        运行模型的 test-time forward pass，为输入特征向量采样 captions。

        在每个 timestep，我们对当前词做 embedding，将它和 previous hidden
        state 传给 RNN 以获得 next hidden state，再用 hidden state 得到所有
        vocab word 的分数，并选择分数最高的词作为 next word。initial hidden
        state 由输入图像特征经过 affine transform 得到，initial word 为
        <START> token。

        对于 LSTM，还需要维护 cell state；此时 initial cell state 应该为 0。

        输入:
        - features: 输入图像特征数组，形状为 (N, D)。
        - max_length: 生成 captions 的最大长度 T。

        返回:
        - captions: 形状为 (N, max_length) 的数组，给出采样得到的 captions，
          其中每个元素都是 [0, V) 范围内的整数。captions 的第一个元素应该为
          第一个采样词，而不是 <START> token。
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # 解包参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO：实现模型的 test-time sampling。需要将输入图像特征经过学到的    #
        # affine transform，初始化 RNN 的 hidden state。传给 RNN 的第一个词    #
        # 应该为 <START> token，其值存储在变量 self._start 中。每个 timestep   #
        # 需要执行以下步骤：                                                   #
        # (1) 使用学到的 word embeddings 对 previous word 做 embedding。        #
        # (2) 使用 previous hidden state 和 embedded current word 执行一个      #
        #     RNN step，得到 next hidden state。                               #
        # (3) 对 next hidden state 应用学到的 affine transformation，得到词表   #
        #     中所有词的分数。                                                 #
        # (4) 选择分数最高的词作为 next word，并把它（word index）写入          #
        #     captions 变量中的相应位置。                                      #
        #                                                                         #
        # 为简单起见，采样到 <END> token 后不需要停止生成；如果愿意，也可以停止。#
        #                                                                         #
        # 提示：不能使用 rnn_forward 或 lstm_forward 函数；需要在循环中调用      #
        # rnn_step_forward 或 lstm_step_forward。                                #
        #                                                                         #
        # 注意：这个函数中仍然是在 minibatch 上工作。如果使用 LSTM，需要把       #
        # 第一个 cell state 初始化为 0。                                         #
        ###########################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return captions
