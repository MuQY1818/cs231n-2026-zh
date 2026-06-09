import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about positions 的 tokens 在 sequence. In
    这个 case, 层 has no learnable 参数, since it is a simple
    函数 的 sines 并 cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct PositionalEncoding 层.

        输入:
         - embed_dim: size 的 embed 维度
         - dropout: dropout 值
         - max_len: maximum possible length 的 输入的 sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an 数组 使用 a "batch 维度" 的 1 (which 将 broadcast
        # across 所有 样本 在 batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO：构造 positional encoding 数组 as described 在            #
        # Transformer_Captioning.ipynb.  goal is 用于 each row 到 alternate     #
        # sine 并 cosine, 并 have exponents 的 0, 0, 2, 2, 4, 4, etc. up 到      #
        # embed_dim. Of course 这个 exact specification is somewhat arbitrary, but #
        # 这个 is what autograder is expecting. For 参考, our solution is #
        # 小于 5 lines 的 code.                                               #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        # Make sure positional encodings 将 be saved 使用 模型
        # 参数 (mostly 用于 completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings 到 输入 sequence.

        输入:
         - x: sequence fed 到 positional encoder 模型, 的 形状
              (N, S, D), 其中 N is batch size, S is sequence length and
              D is embed dim
        返回:
         - 输出: 输入 sequence + positional encodings, 的 形状 (N, S, D)
        """
        N, S, D = x.shape
        # 创建占位变量，后续由你的代码覆盖。
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO：索引到 your 数组 的 positional encodings, 并 add         #
        # appropriate ones 到 输入 sequence. 不要忘记 到 apply dropout    #
        # afterward. This 应该 only take a few lines 的 code.                    #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A 模型 层 which implements a simplified version 的 masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      数据 = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_输出 = attn(query=数据, key=数据, 值=数据)

      # attention 使用 two 输入
      other_数据 = torch.randn(batch_size, sequence_length, embed_dim)
      attn_输出 = attn(query=数据, key=other_数据, 值=other_数据)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention 层.

        输入:
         - embed_dim: 维度 token embedding
         - num_heads: 数量 attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # 我们将 初始化 这些 层 用于 you, since swapping ordering
        # would affect random number generation (and therefore your exact
        # 输出 relative 到 autograder). Note 该 层 使用 a bias
        # term, but 这个 并不是 strictly necessary (and varies by
        # 实现).
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
        Calculate masked attention 输出 用于 provided 数据, 计算
        所有 attention heads 在 并行.

        In 形状 definitions below, N is batch size, S is source
        sequence length, T is target sequence length, 并 E is embedding
        维度.

        输入:
        - query: 输入 数据 被使用 as query, 的 形状 (N, S, E)
        - key: 输入 数据 被使用 as key, 的 形状 (N, T, E)
        - 值: 输入 数据 被使用 as 值, 的 形状 (N, T, E)
        - attn_mask: Array 的 形状 (S, T) 其中 mask[i,j] == 0 indicates token
          i 在 source 应该 not influence token j 在 target.

        返回:
        - 输出: Tensor 的 形状 (N, S, E) giving weighted combination of
          数据 在 值 根据 attention 权重 calculated 使用 key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # 创建占位变量，后续由你的代码覆盖。
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO：实现 multiheaded attention 使用 equations given 在       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want 到 split your 形状 来自 (N, T, E) 到 (N, T, H, E/H),  #
        #     其中 H is 数量 heads.                                      #
        #  2) 函数 torch.matmul 允许 you 到 do a batched 矩阵 multiply.#
        #     For 样本, 你可以 do (N, H, T, E/H) by (N, H, E/H, T) 到 yield a  #
        #     形状 (N, H, T, T). For more 样本, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how 分数 应为 modified 到   #
        #     prevent a 值 来自 influencing 输出. 具体来说, PyTorch   #
        #     函数 masked_fill may come 在 handy.                              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Simple two-层 feed-前向 network 使用 dropout 并 ReLU activation.

        输入:
         - embed_dim: 维度 输入 并 输出 embeddings
         - ffn_dim: Hidden 维度 在 feed前向 network
         - dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        前向传播 用于 feed前向 network.

        输入:
        - x: 输入 tensor 的 形状 (N, T, D)

        返回:
        - out: Output tensor 的 same 形状 as 输入
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    A single 层 的 a Transformer decoder, 被使用 使用 TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        输入:
         - 输入_dim: 数量 expected 特征 在 输入.
         - num_heads: 数量 attention heads
         - dim_feed前向: 维度 feed前向 network 模型.
         - dropout: dropout 值.
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
        Pass 输入 (and mask) through decoder 层.

        输入:
        - tgt: sequence 到 decoder 层, 的 形状 (N, T, D)
        - memory: sequence 来自 last 层 的 encoder, 的 形状 (N, S, D)
        - tgt_mask: parts 的 target sequence 到 mask, 的 形状 (T, T)

        返回:
        - out: Transformer 特征, 的 形状 (N, T, W)
        """

        # Self-attention block (参考 实现)
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO：完成 decoder 层 by implementing remaining two       #
        # sub层: (1) cross-attention block 使用 encoder 输出 as     #
        # memory, 并 (2) feed前向 block. Each block 应该 follow      #
        # same structure as self-attention implemented just above.                 #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    A 层 该 splits an image 到 patches 并 projects each patch 到 an embedding vector.
    使用 as 输入 层 的 a Vision Transformer (ViT).

    输入:
    - img_size: Integer representing height/width 的 输入 image (assumes square image).
    - patch_size: Integer representing height/width 的 each patch (square patch).
    - in_channels: 数量 输入 image channels (e.g., 3 用于 RGB).
    - embed_dim: 维度 linear embedding space.
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

        # Linear projection 的 flattened patches 到 embedding 维度
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        前向传播 用于 patch embedding.

        输入:
        - x: 输入 image tensor 的 形状 (N, C, H, W)

        返回:
        - out: Patch embeddings 使用 形状 (N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size,\
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO：划分 image 到 non-在lapping patches 的 形状             #
        # (C x patch_size x patch_size), 并 rearrange them 到 a tensor 的       #
        # 形状 (N, num_patches, patch_dim). 不要 使用 a for-loop.                #
        # Instead, 你可以 找到 torch.reshape 并 torch.permute helpful 用于 这个   #
        # step. Once patches are flattened, embed them 到 latent vectors     #
        # 使用 projection 层.                                              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return out




class TransformerEncoderLayer(nn.Module):
    """
    A single 层 的 a Transformer encoder, 被使用 使用 TransformerEncoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        输入:
         - 输入_dim: 数量 expected 特征 在 输入.
         - num_heads: 数量 attention heads.
         - dim_feed前向: 维度 feed前向 network 模型.
         - dropout: dropout 值.
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
        Pass 输入 (and mask) through encoder 层.

        输入:
        - src: sequence 到 encoder 层, 的 形状 (N, S, D)
        - src_mask: parts 的 source sequence 到 mask, 的 形状 (S, S)

        返回:
        - out: Transformer 特征, 的 形状 (N, S, D)
        """
        ############################################################################
        # TODO：实现 encoder 层 by applying self-attention followed    #
        # by a feed前向 block. This code 将 be very similar 到 decoder 层. #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return src
