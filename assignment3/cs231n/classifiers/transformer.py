import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions 来自 图像特征 使用 a
    Transformer decoder.

    Transformer receives 输入 vectors 的 size D, has a vocab size 的 V,
    works on sequences 的 length T, 使用s word vectors 的 维度 W, and
    operates on minibatches 的 size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        输入:
        - word_to_idx: A 字典 giving vocabulary. It contains V entries.
          and maps each string 到 a unique integer 在 range [0, V).
        - 输入_dim: Dimension D 的 输入 image 特征 vectors.
        - wordvec_dim: Dimension W 的 word vectors.
        - num_heads: 数量 attention heads.
        - num_层: 数量 transformer 层.
        - max_length: Max possible sequence length.
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
        初始化 权重 网络的.
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
        Given 图像特征 并 caption tokens, return a distribution 在 the
        possible tokens 用于 each timestep. Note 该 since entire sequence
        of captions is provided 所有 at once, we mask out future timesteps.

        输入:
         - 特征: 图像特征, 的 形状 (N, D)
         - captions: ground truth captions, 的 形状 (N, T)

        返回:
         - 分数: score 用于 each token at each timestep, 的 形状 (N, T, V)
        """
        N, T = captions.shape
        # 创建占位变量，后续由你的代码覆盖。
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO：实现 前向 函数 用于 CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have 到 embed your caption 并 add positional              #
        #     encoding. You 然后 have 到 project 图像特征 到 same  #
        #     维度.                                                          #
        #  2) You have 到 prepare a mask (tgt_mask) 用于 masking out future     #
        #     timesteps 在 captions. torch.tril() 函数 可能 help 在 preparing #
        #     这个 mask.                                                           #
        #  3) 最后, apply decoder 特征 on text & image embeddings   #
        #     along 使用 tgt_mask. Project 输出 到 分数 per token      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        Given 图像特征, 使用 greedy decoding 到 predict image caption.

        输入:
         - 特征: 图像特征, 的 形状 (N, D)
         - max_length: maximum possible caption length

        返回:
         - captions: captions 用于 each 样本, 的 形状 (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (其中 所有 tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, 使用 only start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict next token (ignoring 所有 other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose most likely word ID 来自 vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our 在所有 caption 并 our current partial caption.
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    'Produce N identical 层.'
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
    Vision Transformer (ViT) 实现.
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        输入:
         - img_size: Size 的 输入 image (assumed square).
         - patch_size: Size 的 each patch (assumed square).
         - in_channels: 数量 image channels.
         - embed_dim: Embedding 维度 用于 each patch.
         - num_层: 数量 Transformer encoder 层.
         - num_heads: 数量 attention heads.
         - dim_feed前向: Hidden size 的 feed前向 network.
         - num_类别: 数量 分类 标签.
         - dropout: Dropout probability.
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final 分类 层 到 predict 类别分数 来自 pooled token.
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        初始化 权重 网络的.
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
        前向传播 的 Vision Transformer.

        输入:
         - x: 输入 image tensor 的 形状 (N, C, H, W)

        返回:
         - logits: Output 分类 logits 的 形状 (N, num_类别)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # TODO：实现 前向传播 的 Vision Transformer.             #
        # 1. Convert 输入 image 到 a sequence 的 patch vectors.            #
        # 2. Add positional encodings 到 retain spatial information.              #
        # 3. Pass sequence through Transformer encoder.                   #
        # 4. Average pool patch vectors 到 get a 特征 vector 用于 each image.   #
        #    你可以 找到 torch.均值 使用ful.                                      #
        # 5. Feed it through a linear 层 到 produce 类别 logits.              #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################


        return logits
