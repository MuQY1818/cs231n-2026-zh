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

        # 使用 scale 和 shift 调制输出。这是特征融合的一种变体，
        # 比简单相加 feature maps 更强。
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

        # 每一层的 channel 数量，即 [d1, d2, ..., dn]
        dims = [dim] + [dim * m for m in dim_mults]
        # downsampling 路径中每个 U-Net block 的输入和输出
        # e.g. [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # upsampling 路径中每个 U-Net block 的输入和输出
        # e.g. [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # 将 timestep 编码为 context
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # 将 condition（即 text embedding）编码为 context
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # 训练期间丢弃 condition 的概率
        self.uncond_prob = uncond_prob

        # UNet downsampling 和 upsampling blocks.
        # self.downs 是由多个 ModuleList 组成的 ModuleList。
        self.downs = nn.ModuleList([])
        # self.ups 是由多个 ModuleList 组成的 ModuleList。
        self.ups = nn.ModuleList([])

        ####################################################################
        # 下采样 block
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # TODO: 创建一个 UNet downsampling 层 `down_block`，类型为 ModuleList。
            # 它应是包含 3 个 block 的 ModuleList：[ResnetBlock, ResnetBlock, 下采样]。
            # 每个 ResnetBlock 接收 dim_in channels 并输出 dim_in channels。
            # 确保将 context_dim 传给每个 ResnetBlock。
            # 下采样 block 接收 dim_in channels 并输出 dim_out channels。
            # 为了加载 pretrained checkpoint，请严格遵循这个 ModuleList 结构。
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
        # 通过精确镜像 downsampling blocks 来创建 upsampling blocks。
        # self.ups 也会是由多个 ModuleList 组成的 ModuleList。
        # 每个 BlockList 会包含 3 个 block：[上采样, ResnetBlock, ResnetBlock]。
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = None
            ##################################################################
            # TODO: 创建一个 UNet upsampling 层，类型为 ModuleList。
            # 它应是包含 3 个 block 的 ModuleList：[上采样, ResnetBlock, ResnetBlock]。
            # 它会镜像对应的 downsampling block。
            # 不要忘记处理 skip connections：两个 ResnetBlock 的输入通道数
            # 都应包含 2 x dim_out channels。
            ##################################################################

            self.ups.append(up_block)
            ##################################################################

        # 最终卷积，用于映射到输出通道
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """Classifier-free guidance 前向传播。model_kwargs 应包含 `cfg_scale`。"""

        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        ##################################################################
        # TODO：根据以下论文的公式 (6) 应用 classifier-free guidance：
        # https://arxiv.org/pdf/2207.12598 i.e.
        # x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        #
        # 你需要调用两次 self.forward。
        # 对于 unconditional sampling，在 `text_emb` 中传入 None。
        ##################################################################

        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """通过 U-Net 的前向传播。
        参数:
            x: 输入 tensor，形状为 (batch_size, channels, height, width).
            time: time steps tensor，形状为 (batch_size,).
            model_kwargs: 包含额外模型输入的字典，包括形状为
                (batch_size, condition_dim) 的 "text_emb"（text embedding）。

        返回:
            x: 输出 tensor，形状为 (batch_size, channels, height, width).
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
        # TODO：在 context 条件下通过 U-Net 处理 `x`。
        #
        # 1. Downsampling:
        #    - 使用 context 依次通过每个 downsampling block 处理 `x`。
        #    - 每个 ResNet block 之后，将输出（feature maps）保存到 list 或 dict，
        #      以便在 upsampling path 中作为 skip connections 使用。
        #    - 确保将 context 传入每个 ResNet block。
        #
        # 2. Middle:
        #    - 使用 context 通过 middle blocks 处理 `x`。
        #
        # 3. Upsampling:
        #    - 使用 context 通过每个 upsampling block 处理 `x`。
        #    - 在每个 ResNet block 之前，将输入与来自 downsampling path 的
        #      对应 skip connection 拼接。
        #    - 确保将 context 传入每个 ResNet block。
        ##################################################################

        ##################################################################

        # Final block
        x = self.final_conv(x)

        return x
