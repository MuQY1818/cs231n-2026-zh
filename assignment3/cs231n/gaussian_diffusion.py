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
        register_buffer("betas", betas)  # 可以 be accessed as self.betas
        register_buffer("alphas", alphas)  # 可以 be accessed as self.alphas
        register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod

        #############################################################################
        # Other coefficients 需要 到 transform between x_t, x_0, 并 noise
        # Note 该 根据 Eq. (4) 并 its reparameterization 在 Eq. (14),
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # 其中 noise is sampled 来自 N(0, 1)
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
        # For posterior q(x_{t-1} | x_t, x_0) 根据 Eq. (6) 并 (7) 的 paper.
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
        """Get x_start 来自 x_t 并 noise 根据 Eq. (14) 的 paper.
        参数:
            x_t: (b, *) tensor. 带噪图像.
            t: (b,) tensor. 时间步.
            noise: (b, *) tensor. Noise 来自 N(0, 1).
        返回:
            x_start: (b, *) tensor. 起始图像.
        """
        x_start = None
        ####################################################################
        # TODO：
        # Transform x_t 并 noise 到 get x_start 根据 Eq.(4) 并 Eq.(14).
        # Look at coeffs 在 `__init__` method 并 使用 `extract` 函数.
        ####################################################################

        ####################################################################
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """Get noise 来自 x_t 并 x_start 根据 Eq. (14) 的 paper.
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
        # Transform x_t 并 noise 到 get x_start 根据 Eq.(4) 并 Eq.(14).
        # Look at coeffs 在 `__init__` method 并 使用 `extract` 函数.
        ####################################################################

        ####################################################################
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        """Get posterior q(x_{t-1} | x_t, x_0) 根据 Eq. (6) 并 (7) 的 paper.
        参数:
            x_start: (b, *) tensor. Predicted start image.
            x_t: (b, *) tensor. 带噪图像.
            t: (b,) tensor. 时间步.
        返回:
            posterior_均值: (b, *) tensor. Mean 的 posterior.
            posterior_标准差: (b, *) tensor. Std 的 posterior.
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
        """Sample 来自 p(x_{t-1} | x_t) 根据 Eq. (6) 的 paper. 使用 only during inference.
        参数:
            x_t: (b, *) tensor. 带噪图像.
            t: int. Sampling time step.
            模型_kwargs: additional arguments 用于 模型.
        返回:
            x_tm1: (b, *) tensor. Sampled image.
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,)
        x_tm1 = None  # sample x_{t-1} 来自 p(x_{t-1} | x_t)

        ##################################################################
        # TODO：实现 sampling step p(x_{t-1} | x_t) 根据 Eq. (6):
        #
        # - Steps:
        #   1. Get 模型 预测 by 调用 self.模型 使用 appropriate args.
        #   2. 模型 输出 可以 be either noise or x_start depending on self.objective.
        #      你可以 恢复 other by 调用 self.predict_start_来自_noise or
        #      self.predict_noise_来自_start as 需要.
        #   3. Clamp 预测的 x_start 到 valid range [-1, 1]. This ensures the
        #      generation remains stable during denoising iterations.
        #   4. Get 均值 并 标准差 用于 q(x_{t-1} | x_t, x_0) 使用 self.q_posterior,
        #      并 sample x_{t-1}.
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
        """Sample 来自 q(x_t | x_0) 根据 Eq. (4) 的 paper.

        参数:
            x_start: (b, *) tensor. 起始图像.
            t: (b,) tensor. 时间步.
            noise: (b, *) tensor. Noise 来自 N(0, 1).
        返回:
            x_t: (b, *) tensor. 带噪图像.
        """

        x_t = None
        ####################################################################
        # TODO：
        # Implement sampling 来自 q(x_t | x_0) 根据 Eq. (4) 的 paper.
        # Hints: (1) Look at `__init__` method 到 see pre计算得到的 coefficients.
        # (2) 使用 `extract` 函数 defined above 到 extract coefficients
        # for given time step `t`. (3) Re调用 该 sampling 来自 N(mu, sigma^2)
        # 可以 be done as: x_t = mu + sigma * noise 其中 noise is sampled 来自 N(0, 1).
        # Approximately 3 lines 的 code.
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
        # Implement 损失 函数 根据 Eq. (14) 的 paper.
        # First, sample x_t 来自 q(x_t | x_0) 使用 `q_sample` 函数.
        # Then, get 模型 预测 by 调用 self.模型 使用 appropriate args.
        # 最后, 计算 weighted MSE 损失.
        # Approximately 3-4 lines 的 code.
        ####################################################################

        ####################################################################

        return loss


def extract(a, t, x_shape):
    """
    Extracts appropriate coefficient 值 基于 given timesteps.

    This 函数 gathers 值 来自 coefficient tensor `a` 根据
    given timesteps `t` 并 reshapes them 到 match required 形状 such 该
    it supports broadcasting 使用 tensor 的 given 形状 `x_形状`.

    参数:
        a (torch.Tensor): A tensor 的 形状 (T,), containing coefficient 值 用于 所有 timesteps.
        t (torch.Tensor): A tensor 的 形状 (b,), representing timesteps 用于 each sample 在 batch.
        x_形状 (tuple): 形状 的 输入 image tensor, usu所有y (b, c, h, w).

    返回:
        torch.Tensor: A tensor 的 形状 (b, 1, 1, 1), containing extracted coefficient 值
                      来自 a 用于 corresponding timestep 的 each batch element, reshaped 根据ly.
    """
    b, *_ = t.shape  # Extract batch size 来自 timestep tensor
    out = a.gather(-1, t)  # Gather coefficient 值 来自 `a` 基于 `t`
    out = out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # reshape 到 (b, 1, 1, 1) 用于 broadcasting
    return out


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed 在 original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed 在 https://openreview.net/forum?id=-NEXDKk8gZ
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
    proposed 在 https://arxiv.org/abs/2212.11972 - Figure 8
    better 用于 images > 64x64, when 使用 during 训练
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
