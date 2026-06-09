import torch
import torch.nn as nn
from tqdm.auto import tqdm
import math


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x_start",
        }, "objective must be either pred_noise (predict noise) or pred_x_start (predict image start)"

        # 辅助函数：把一些常量注册为 buffer，以确保
        # 它们与模型参数位于同一设备。
        # See https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # 每个 buffer 可通过 `self.name` 访问
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        #############################################################################
        # noise schedule 中的 beta 和 alpha 值
        #############################################################################
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)  # 可通过 self.betas 访问
        register_buffer("alphas", alphas)  # 可通过 self.alphas 访问
        register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod

        #############################################################################
        # 其他系数，用于在 x_t、x_0 和 noise 之间转换。
        # 注意，根据公式 (4) 及其在公式 (14) 中的重参数化，
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # 其中 noise 从 N(0, 1) 采样。
        #############################################################################
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        # register_buffer(
        #     "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        # )

        #############################################################################
        # 根据论文公式 (6) 和 (7)，用于 posterior q(x_{t-1} | x_t, x_0)。
        #############################################################################
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        #################################################################
        # 损失权重
        #################################################################
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        return img * 2 - 1

    def unnormalize(self, img):
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t, t, noise):
        """根据论文公式 (14)，由 x_t 和 noise 得到 x_start。
        参数:
            x_t: (b, *) tensor. 带噪图像.
            t: (b,) tensor. 时间步.
            noise: (b, *) tensor. 来自 N(0, 1) 的 noise.
        返回:
            x_start: (b, *) tensor. 起始图像.
        """
        x_start = None
        ####################################################################
        # TODO：
        # 根据公式 (4) 和公式 (14)，由 x_t 和 noise 得到 x_start。
        # 查看 `__init__` 方法中的系数，并使用 `extract` 函数。
        ####################################################################

        ####################################################################
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """根据论文公式 (14)，由 x_t 和 x_start 得到 noise。
        参数:
            x_t: (b, *) tensor. 带噪图像.
            t: (b,) tensor. 时间步.
            x_start: (b, *) tensor. 起始图像.
        返回:
            pred_noise: (b, *) tensor. 预测噪声.
        """
        pred_noise = None
        ####################################################################
        # TODO：
        # 根据公式 (4) 和公式 (14)，由 x_t 和 x_start 得到 noise。
        # 查看 `__init__` 方法中的系数，并使用 `extract` 函数。
        ####################################################################

        ####################################################################
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        """根据论文公式 (6) 和 (7)，得到 posterior q(x_{t-1} | x_t, x_0)。
        参数:
            x_start: (b, *) tensor. 预测的起始图像.
            x_t: (b, *) tensor. 带噪图像.
            t: (b,) tensor. 时间步.
        返回:
            posterior_mean: (b, *) tensor. posterior 的 mean.
            posterior_std: (b, *) tensor. posterior 的 std.
        """
        posterior_mean = None
        posterior_std = None
        ####################################################################
        # 我们已经为你实现了这个方法。
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        ####################################################################
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        """根据论文公式 (6) 从 p(x_{t-1} | x_t) 采样，仅在 inference 时使用。
        参数:
            x_t: (b, *) tensor. 带噪图像.
            t: int. 采样时间步.
            model_kwargs: 传给模型的额外参数.
        返回:
            x_tm1: (b, *) tensor. 采样得到的图像.
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,)
        x_tm1 = None  # sample x_{t-1} 来自 p(x_{t-1} | x_t)

        ##################################################################
        # TODO：根据公式 (6) 实现 sampling step p(x_{t-1} | x_t):
        #
        # - 步骤：
        #   1. 使用合适的参数调用 self.model，得到模型预测。
        #   2. 模型输出可能是 noise，也可能是 x_start，取决于 self.objective。
        #      你可以按需调用 self.predict_start_from_noise 或
        #      self.predict_noise_from_start 来恢复另一个量。
        #   3. 将预测的 x_start clamp 到有效范围 [-1, 1]，以保证 denoising
        #      迭代过程中的生成稳定性。
        #   4. 使用 self.q_posterior 得到 q(x_{t-1} | x_t, x_0) 的 mean 和 std，
        #      并采样 x_{t-1}。
        ##################################################################
        
        ##################################################################

        return x_tm1

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def q_sample(self, x_start, t, noise):
        """根据论文公式 (4)，从 q(x_t | x_0) 采样。

        参数:
            x_start: (b, *) tensor. 起始图像.
            t: (b,) tensor. 时间步.
            noise: (b, *) tensor. 来自 N(0, 1) 的 noise.
        返回:
            x_t: (b, *) tensor. 带噪图像.
        """

        x_t = None
        ####################################################################
        # TODO：
        # 根据论文公式 (4) 实现从 q(x_t | x_0) 的采样。
        # 提示：(1) 查看 `__init__` 方法中的预计算系数。
        # (2) 使用上面定义的 `extract` 函数提取给定时间步 `t` 的系数。
        # (3) 记住从 N(mu, sigma^2) 采样可写成：
        #     x_t = mu + sigma * noise，其中 noise 从 N(0, 1) 采样。
        # 大约需要 3 行代码。
        ####################################################################

        ####################################################################
        return x_t

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
        x_start = self.normalize(x_start)  # (b, *)
        noise = torch.randn_like(x_start)  # (b, *)
        target = noise if self.objective == "pred_noise" else x_start  # (b, *)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)
        loss = None

        ####################################################################
        # TODO：
        # 根据论文公式 (14) 实现损失函数。
        # 首先使用 `q_sample` 函数从 q(x_t | x_0) 采样 x_t。
        # 然后用合适参数调用 self.model，得到模型预测。
        # 最后计算加权 MSE 损失。
        # 大约需要 3-4 行代码。
        ####################################################################

        ####################################################################

        return loss


def extract(a, t, x_shape):
    """
    根据给定 timesteps 提取相应的 coefficient 值。

    此函数会根据给定 timesteps `t`，从 coefficient tensor `a` 中 gather 值，
    并 reshape 成所需形状，使其可以与给定形状 `x_shape` 的 tensor 广播。

    参数:
        a (torch.Tensor): 形状为 (T,) 的 tensor，包含所有 timesteps 的 coefficient 值.
        t (torch.Tensor): 形状为 (b,) 的 tensor，表示 batch 中每个样本的 timestep.
        x_shape (tuple): 输入 image tensor 的形状，通常为 (b, c, h, w).

    返回:
        torch.Tensor: 形状为 (b, 1, 1, 1) 的 tensor，包含从 `a` 中为每个
                      batch 元素对应 timestep 提取并 reshape 后的 coefficient 值.
    """
    b, *_ = t.shape  # 从 timestep tensor 中提取 batch size
    out = a.gather(-1, t)  # 根据 `t` 从 `a` 中 gather coefficient 值
    out = out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # reshape 为 (b, 1, 1, 1) 用于 broadcasting
    return out


def linear_beta_schedule(timesteps):
    """
    linear schedule，由原始 DDPM 论文提出。
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    提出于 https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    提出于 https://arxiv.org/abs/2212.11972 - Figure 8
    训练时对 images > 64x64 通常更好。
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
