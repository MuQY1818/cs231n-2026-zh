import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
此文件定义 transformer 中常用的 layer 类型。
"""

class PositionalEncoding(nn.Module):
    """
    编码序列中 token 的位置信息。这里的层没有可学习参数，因为它只是
    由 sine 和 cosine 构成的简单函数。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        构造 PositionalEncoding 层。

        输入:
         - embed_dim: embedding 维度大小
         - dropout: dropout 值
         - max_len: 输入序列的最大可能长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # 创建一个 batch 维度为 1 的数组，该维度会广播到 batch 中的所有样本。
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO：按照 Transformer_Captioning.ipynb 中的描述构造 positional         #
        # encoding 数组。目标是让每一行交替使用 sine 和 cosine，并让指数为       #
        # 0, 0, 2, 2, 4, 4, ...，一直到 embed_dim。这个具体设定有些任意，        #
        # 但 autograder 会按此检查。作为参考，我们的解法少于 5 行代码。          #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # 确保 positional encoding 会随模型参数一起保存，主要是为了完整性。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将 positional embedding 按元素加到输入序列上。

        输入:
         - x: 送入 positional encoder 的序列，形状为 (N, S, D)，其中
              N 是 batch size，S 是 sequence length，D 是 embed dim
        返回:
         - output: 输入序列加上 positional encoding，形状为 (N, S, D)
        """
        N, S, D = x.shape
        # 创建占位变量，后续由你的代码覆盖。
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO：索引出需要的 positional encoding，并加到输入序列上。             #
        # 不要忘记随后应用 dropout。这里应该只需要几行代码。                    #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    实现 masked attention 简化版本的模型层，出自
    "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)。

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention with two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        构造一个新的 MultiHeadAttention 层。

        输入:
         - embed_dim: token embedding 的维度
         - num_heads: attention head 的数量
         - dropout: Dropout 概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # 我们已经为你初始化这些层，因为更换顺序会影响随机数生成，
        # 从而影响与 autograder 对比时的精确输出。注意这些层使用 bias，
        # 但 bias 并非严格必要，不同实现可能不同。
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        根据给定数据计算 masked attention 输出，并行计算所有 attention head。

        在下面的形状定义中，N 是 batch size，S 是 source sequence length，
        T 是 target sequence length，E 是 embedding 维度。

        输入:
        - query: 作为 query 的输入数据，形状为 (N, S, E)
        - key: 作为 key 的输入数据，形状为 (N, T, E)
        - value: 作为 value 的输入数据，形状为 (N, T, E)
        - attn_mask: 形状为 (S, T) 的数组，其中 mask[i, j] == 0 表示
          source 中的 token i 不应影响 target 中的 token j。

        返回:
        - output: 形状为 (N, S, E) 的 tensor，表示根据 key 和 query
          计算出的 attention 权重对 value 数据做加权组合。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # 创建占位变量，后续由你的代码覆盖。
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO：使用 Transformer_Captioning.ipynb 中给出的公式实现               #
        # multi-headed attention。                                                #
        # 提示：                                                                  #
        #  1) 需要把形状从 (N, T, E) 拆成 (N, T, H, E/H)，其中 H 是 head 数量。  #
        #  2) torch.matmul 支持 batched matrix multiply。例如，                  #
        #     (N, H, T, E/H) 与 (N, H, E/H, T) 相乘会得到 (N, H, T, T)。        #
        #     更多示例见：                                                       #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) 应用 attn_mask 时，思考如何修改分数来阻止某个 value 影响输出。     #
        #     PyTorch 的 masked_fill 函数可能有用。                              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        简单的两层 feed-forward network，使用 dropout 和 ReLU activation。

        输入:
         - embed_dim: 输入和输出 embedding 的维度
         - ffn_dim: feed-forward network 的隐藏维度
         - dropout: Dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        feed-forward network 的前向传播。

        输入:
        - x: 输入 tensor，形状为 (N, T, D)

        返回:
        - out: 输出 tensor，形状与输入相同
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder 中的单层，用于 TransformerDecoder。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构造 TransformerDecoderLayer 实例。

        输入:
         - input_dim: 输入中特征的期望数量。
         - num_heads: attention head 的数量。
         - dim_feedforward: feed-forward network 的维度。
         - dropout: dropout 值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None):
        """
        将输入和 mask 送入 decoder 层。

        输入:
        - tgt: 送入 decoder 层的序列，形状为 (N, T, D)
        - memory: 来自 encoder 最后一层的序列，形状为 (N, S, D)
        - tgt_mask: target sequence 中需要 mask 的部分，形状为 (T, T)

        返回:
        - out: Transformer 特征，形状为 (N, T, W)
        """

        # Self-attention block（参考实现）
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO：实现剩余两个子层来完成 decoder 层：                              #
        # (1) 使用 encoder 输出作为 memory 的 cross-attention block；             #
        # (2) feed-forward block。每个 block 都应遵循上面 self-attention          #
        # 的相同结构。                                                           #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    将图像切分成 patches，并把每个 patch 投影为 embedding vector 的层。
    作为 Vision Transformer (ViT) 的输入层。

    输入:
    - img_size: 输入图像的高/宽，假设为正方形。
    - patch_size: 每个 patch 的高/宽，假设为正方形。
    - in_channels: 输入图像通道数，例如 RGB 为 3。
    - embed_dim: linear embedding space 的维度。
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # 将展平后的 patches 线性投影到 embedding 维度。
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        patch embedding 的前向传播。

        输入:
        - x: 输入 image tensor，形状为 (N, C, H, W)

        返回:
        - out: Patch embeddings，形状为 (N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size,\
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO：将 image 划分为形状为 (C x patch_size x patch_size) 的          #
        # non-overlapping patches，并重排成形状为 (N, num_patches, patch_dim)  #
        # 的 tensor。不要使用 for-loop。torch.reshape 和 torch.permute          #
        # 可能对这一步有帮助。patch 展平后，使用 projection 层将其嵌入为        #
        # latent vector。                                                       #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return out




class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder 中的单层，用于 TransformerEncoder。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构造 TransformerEncoderLayer 实例。

        输入:
         - input_dim: 输入中特征的期望数量。
         - num_heads: attention head 的数量。
         - dim_feedforward: feed-forward network 的维度。
         - dropout: dropout 值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        将输入和 mask 送入 encoder 层。

        输入:
        - src: 送入 encoder 层的序列，形状为 (N, S, D)
        - src_mask: source sequence 中需要 mask 的部分，形状为 (S, S)

        返回:
        - out: Transformer 特征，形状为 (N, S, D)
        """
        ############################################################################
        # TODO：依次应用 self-attention 和 feed-forward block 来实现 encoder 层。 #
        # 这段代码会和 decoder 层非常相似。                                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return src
