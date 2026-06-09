import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions 来自 图像特征 使用 a recurrent
    neural network.

    RNN receives 输入 vectors 的 size D, has a vocab size 的 V, works on
    sequences 的 length T, has an RNN hidden 维度 H, 使用s word vectors
    of 维度 W, 并 operates on minibatches 的 size N.

    Note 该 we don't 使用 any 正则化 用于 CaptioningRNN.
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
        Construct a new CaptioningRNN instance.

        输入:
        - word_to_idx: A 字典 giving vocabulary. It contains V entries,
          and maps each string 到 a unique integer 在 range [0, V).
        - 输入_dim: Dimension D 的 输入 image 特征 vectors.
        - wordvec_dim: Dimension W 的 word vectors.
        - hidden_dim: Dimension H 用于 hidden state 的 RNN.
        - cell_type: What type 的 RNN 到 使用; either 'rnn' or 'lstm'.
        - dtype: numpy 数据type 到 使用; 使用 float32 用于训练 并 float64 for
          numeric 梯度 checking.
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

        # 初始化 hidden 到词表输出的权重
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # 将参数转换为正确 dtype
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        计算 训练时 损失 用于 RNN. We 输入 图像特征 and
        ground-truth captions 用于 those images, 并 使用 an RNN (or LSTM) 到 计算
        损失 并 梯度 on 所有 参数.

        输入:
        - 特征: 输入 图像特征, 的 形状 (N, D)
        - captions: Ground-truth captions; an integer 数组 的 形状 (N, T + 1) 其中
          each element is 在 range 0 <= y[i, t] < V

        返回 a tuple of:
        - 损失: Scalar 损失
        """
        # Cut captions 到 two pieces: captions_in has everything but last word
        # and 将 be 输入 到 RNN; captions_out has everything but first
        # word 并 这个 is what 我们将 expect RNN 到 generate. These are offset
        # by one relative 到 each other beca使用 RNN 应该 produce word (t+1)
        # after receiving word t. first element 的 captions_in 将 be START
        # token, 并 first element 的 captions_out 将 be first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll 需要 这个
        mask = captions_out != self._null

        # Weight 并 bias 用于 affine transform 来自 图像特征 到 initial
        # hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding 矩阵
        W_embed = self.params["W_embed"]

        # 输入-to-hidden, hidden-to-hidden, 并 偏置 用于 RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight 并 bias 用于 hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO：实现 前向传播 用于 CaptioningRNN.                  #
        # In 前向传播 你需要 到 do following:                   #
        # (1) 使用 an affine transformation 到 计算 initial hidden state     #
        #     来自 图像特征. This 应该 produce an 数组 的 形状 (N, H)#
        # (2) 使用 a word embedding 层 到 transform words 在 captions_in     #
        #     来自 indices 到 vectors, giving an 数组 的 形状 (N, T, W).         #
        # (3) 使用 either a vanilla RNN or LSTM (depending on self.cell_type) 到    #
        #     process sequence 的 输入 word vectors 并 produce hidden state  #
        #     vectors 用于 所有 timesteps, producing an 数组 的 形状 (N, T, H).    #
        # (4) 使用 a (temporal) affine transformation 到 计算 分数 在    #
        #     vocabulary at every timestep 使用 hidden states, giving an      #
        #     数组 的 形状 (N, T, V).                                            #
        # (5) 使用 (temporal) softmax 到 计算 损失 使用 captions_out, ignoring  #
        #     点 其中 输出 word is <NULL> 使用 mask above.     #
        #                                                                          #       
        # Please ensure 该 your 实现 is agnostic 的 输入 tensors  #
        # 数据类型.                                                              #
        #                                                                          #
        # 不要 worry about regularizing 权重 or their 梯度!          #
        #                                                                          #
        # You also don't have 到 implement 反向传播.                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        Run a 测试时 前向传播 用于 模型, sampling captions 用于 输入
        特征 vectors.

        At each timestep, we embed current word, pass it 并 previous hidden
        state 到 RNN 到 get next hidden state, 使用 hidden state 到 get
        分数 用于 所有 vocab words, 并 choose word 使用 highest score as
        next word. initial hidden state is 计算得到的 by applying an affine
        transform 到 输入 图像特征, 并 initial word is <START>
        token.

        For LSTMs you 将 also have 到 keep track 的 cell state; 在 该 case
        initial cell state 应为 zero.

        输入:
        - 特征: Array 的 输入 图像特征 的 形状 (N, D).
        - max_length: Maximum length T 的 generated captions.

        返回:
        - captions: Array 的 形状 (N, max_length) giving sampled captions,
          其中 each element is an integer 在 range [0, V). first element
          of captions 应为 first sampled word, not <START> token.
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # 解包参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO：实现 测试时 sampling 用于 模型. 你需要 到      #
        # 初始化 hidden state 的 RNN by applying learned affine   #
        # transform 到 输入 图像特征. first word 该 you feed 到  #
        # RNN 应为 <START> token; its 值 is 存储 在         #
        # 变量 self._start. At each timestep 你需要 到 do to:          #
        # (1) Embed previous word 使用 learned word embeddings           #
        # (2) Make an RNN step 使用 previous hidden state 并 embedded   #
        #     current word 到 get next hidden state.                          #
        # (3) Apply learned affine transformation 到 next hidden state 到 #
        #     get 分数 用于 所有 words 在 vocabulary                          #
        # (4) Select word 使用 highest score as next word, writing it #
        #     (word index) 到 appropriate slot 在 captions 变量   #
        #                                                                         #
        # 为简单起见, you 不要 需要 到 stop generating after an <END> token #
        # is sampled, but 你可以 if you want to.                                 #
        #                                                                         #
        # 提示： You 将 not be able 到 使用 rnn_前向 or lstm_前向       #
        # 函数; you'll 需要 到 调用 rnn_step_前向 or lstm_step_前向 在 #
        # a loop.                                                                 #
        #                                                                         #
        # 注意： we are still working 在 minibatches 在 这个 函数. Also if   #
        # you are 使用 an LSTM, 初始化 first cell state 到 zeros.        #
        ###########################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return captions
