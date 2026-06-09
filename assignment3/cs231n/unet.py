import copy
from einops import rearrange
from torch import einsum

from torch import nn
import torch
import torch.nn.functional as F
import math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    """将图像特征分辨率上采样 2 倍。"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """将图像特征分辨率下采样 2 倍。"""
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    """RMSNorm 层，是一种计算高效的 LayerNorm 简化变体。"""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    """用于时间步的正弦位置嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """带特征调制的卷积 block。"""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # Scale 并 平移 are 使用 到 modulate 输出. This is a variant
        # of 特征 fusion, more powerful than simply adding 特征 maps.
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """带上下文相关特征调制的类 ResNet block。"""

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):

        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context)
            context = rearrange(context, "b c -> b c 1 1")
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        # 数量 channels at each 层 i.e. [d1, d2, ..., dn]
        dims = [dim] + [dim * m for m in dim_mults]
        # 输入 并 输出 用于 each U-Net block 在 downsampling 层
        # e.g. [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # 输入 并 输出 用于 each U-Net block 在 upsampling 层
        # e.g. [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # Encoding timestep as context
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Encoding condition (i.e. text embedding) as context
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Probability 的 dropping condition during 训练
        self.uncond_prob = uncond_prob

        # UNet downsampling 并 upsampling blocks.
        # self.downs is a ModuleList 的 ModuleLists.
        self.downs = nn.ModuleList([])
        # self.ups is a ModuleList 的 ModuleLists.
        self.ups = nn.ModuleList([])

        ####################################################################
        # 下采样 block
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # TODO: Create one UNet downsampling 层 `down_block` as a ModuleList.
            # It 应为 a ModuleList 的 3 blocks [ResnetBlock, ResnetBlock, 下采样].
            # Each ResnetBlock operates on dim_in channels 并 输出 dim_in channels.
            # Make sure 到 pass context_dim 到 each ResnetBlock.
            # 下采样 block operates on dim_in channels 并 输出 dim_out channels.
            # Make sure 到 exactly follow 这个 structure 的 ModuleList 在 order to
            # load a pre训练ed check点.
            ##################################################################

            ##################################################################
            self.downs.append(down_block)

        # 中间 block
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        ####################################################################
        # 上采样 block
        ####################################################################
        # Create upsampling blocks by exactly mirroring downsampling blocks.
        # self.ups 将 also be a ModuleList 的 ModuleLists.
        # Each BlockList 将 contain 3 blocks [上采样, ResnetBlock, ResnetBlock].
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = None
            ##################################################################
            # TODO: Create one UNet upsampling 层 as a ModuleList.
            # It 应为 a ModuleList 的 3 blocks [上采样, ResnetBlock, ResnetBlock].
            # This 将 mirror corresponding downsampling block.
            # 不要忘记 到 account 用于 skip connections by having 2 x dim_out
            # channels at 输入 的 both ResnetBlocks.
            ##################################################################

            self.ups.append(up_block)
            ##################################################################

        # 最终卷积，用于映射到输出通道
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """Classifier-free guidance 前向传播。模型_kwargs 应包含 `cfg_缩放`。"""

        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        ##################################################################
        # TODO：应用 分类器-free guidance 使用 Eq. (6) 来自
        # https://arxiv.org/pdf/2207.12598 i.e.
        # x = (缩放 + 1) * eps(x_t, cond) - 缩放 * eps(x_t, empty)
        #
        # You 将 have 到 调用 self.前向 two times.
        # For unconditional sampling, pass None in`text_emb`.
        ##################################################################

        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """通过 U-Net 的前向传播。
        参数:
            x: 输入 tensor 的 形状 (batch_size, channels, height, width).
            time: Tensor 的 time steps 的 形状 (batch_size,).
            模型_kwargs: A 字典 的 additional 模型 输入 including
                "text_emb" (text embedding) 的 形状 (batch_size, condition_dim).

        返回:
            x: Output tensor 的 形状 (batch_size, channels, height, width).
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        # 嵌入时间步
        context = self.time_mlp(time)

        # 嵌入条件并加入 context
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
        if self.training:
            # 随机丢弃条件
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # B x 1
            cond_emb = cond_emb * mask
        context = context + self.condition_mlp(cond_emb)

        # 初始卷积
        x = self.init_conv(x)

        ##################################################################
        # TODO：处理 `x` through U-Net conditioned on context.
        #
        # 1. Downsampling:
        #    - Process `x` through each downsampling block 使用 context.
        #    - After each ResNet block, save 输出 (特征 maps) 在 a list or dict
        #      用于 使用 as skip connections 在 upsampling path.
        #    - Make sure 到 pass context 到 each ResNet block.
        #
        # 2. Middle:
        #    - Process `x` through middle blocks 使用 context.
        #
        # 3. Upsampling:
        #    - Process `x` through each upsampling block 使用 context.
        #    - Before each ResNet block, concatenate 输入 使用 corresponding
        #      skip connection 来自 downsampling path.
        #    - Make sure 到 pass context 到 each ResNet block.
        ##################################################################

        ##################################################################

        # Final block
        x = self.final_conv(x)

        return x
