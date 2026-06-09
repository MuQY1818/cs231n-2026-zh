import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    CaptioningTransformer 使用 Transformer decoder 根据图像特征生成 caption。

    Transformer 接收维度为 D 的输入向量，词表大小为 V，处理长度为 T 的序列，
    使用维度为 W 的 word vector，并按大小为 N 的 minibatch 运行。
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        构造一个新的 CaptioningTransformer 实例。

        输入:
        - word_to_idx: 给出词表的字典，包含 V 个条目，并将每个字符串映射到
          区间 [0, V) 内唯一的整数。
        - input_dim: 输入图像特征向量的维度 D。
        - wordvec_dim: word vector 的维度 W。
        - num_heads: attention head 的数量。
        - num_layers: transformer 层数。
        - max_length: 最大可能序列长度。
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        初始化网络权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        给定图像特征和 caption token，返回每个时间步上的 token 分布。
        由于会一次性提供完整 caption 序列，需要 mask 掉未来时间步。

        输入:
         - features: 图像特征，形状为 (N, D)
         - captions: ground truth captions，形状为 (N, T)

        返回:
         - scores: 每个时间步上各 token 的分数，形状为 (N, T, V)
        """
        N, T = captions.shape
        # 创建占位变量，后续由你的代码覆盖。
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO：实现 CaptionTransformer 的 forward 函数。                         #
        # 提示：                                                                  #
        #  1) 先对 caption 做 embedding 并加入 positional encoding；随后将图像   #
        #     特征投影到相同维度。                                                #
        #  2) 准备一个 mask (tgt_mask) 来屏蔽 caption 中的未来时间步。            #
        #     torch.tril() 可能有助于构造这个 mask。                              #
        #  3) 最后，将 text embedding 和 image embedding 连同 tgt_mask 输入       #
        #     decoder，并把输出投影为每个 token 的分数。                          #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        给定图像特征，使用 greedy decoding 预测 image caption。

        输入:
         - features: 图像特征，形状为 (N, D)
         - max_length: 最大可能 caption 长度

        返回:
         - captions: 每个样本的 captions，形状为 (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # 创建一个空 captions tensor，其中所有 token 都是 NULL。
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # 创建只包含 start token 的 partial caption。
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # 预测下一个 token，忽略其他时间步。
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # 从词表中选择概率最高的 word ID。
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # 更新完整 caption 和当前 partial caption。
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    '生成 N 个相同的层。'
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output



class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 实现。
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        输入:
         - img_size: 输入图像大小，假设为正方形。
         - patch_size: 每个 patch 的大小，假设为正方形。
         - in_channels: 图像通道数。
         - embed_dim: 每个 patch 的 embedding 维度。
         - num_layers: Transformer encoder 层数。
         - num_heads: attention head 的数量。
         - dim_feedforward: feed-forward network 的隐藏层大小。
         - num_classes: 分类标签数量。
         - dropout: Dropout 概率。
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终分类层，根据 pooled token 预测类别分数。
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        初始化网络权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Vision Transformer 的前向传播。

        输入:
         - x: 输入 image tensor，形状为 (N, C, H, W)

        返回:
         - logits: 输出分类 logits，形状为 (N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # TODO：实现 Vision Transformer 的前向传播。                              #
        # 1. 将输入图像转换为 patch vector 序列。                                 #
        # 2. 加入 positional encoding，以保留空间信息。                           #
        # 3. 将序列送入 Transformer encoder。                                     #
        # 4. 对 patch vector 做 average pooling，得到每张图像的特征向量。         #
        #    torch.mean 可能有用。                                                #
        # 5. 将特征向量送入线性层，生成类别 logits。                              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################


        return logits
