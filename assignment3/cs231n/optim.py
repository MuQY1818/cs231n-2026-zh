"""
This file implements various first-order update rules 该 are commonly 使用
用于训练 neural networks. Each update rule accepts current 权重 并 the
梯度 的 损失 使用 respect 到 those 权重 并 produces next set of
权重. Each update rule has same interface:

def update(w, dw, config=None):

输入:
  - w: A numpy 数组 giving current 权重.
  - dw: A numpy 数组 的 same 形状 as w giving 梯度 的 the
    损失 使用 respect 到 w.
  - config: A 字典 containing hyperparameter 值 such as learning
    rate, momentum, etc. If update rule requires caching 值 在 many
    iterations, 然后 config 将 also hold 这些 cached 值.

返回:
  - next_w: next 点 after update.
  - config: config 字典 到 be passed 到 next iteration 的 the
    update rule.

注意： For most update rules, default 学习率 将 probably not
perform well; however default 值 的 other hyper参数 应该
work well 用于 a variety 的 different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal 到 w."""

import numpy as np


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic 梯度 descent.

    config format:
    - learning_rate: Scalar 学习率.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic 梯度 descent 使用 momentum.

    config format:
    - learning_rate: Scalar 学习率.
    - momentum: Scalar between 0 并 1 giving momentum 值.
      Setting momentum = 0 reduces 到 sgd.
    - velocity: A numpy 数组 的 same 形状 as w 并 dw 使用 到 存储 a
      moving average 的 梯度.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    v *= config["momentum"]
    v -= config["learning_rate"] * dw
    w += v
    next_w = w
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用s RMSProp update rule, which 使用s a moving average 的 squared
    梯度 值 到 set adaptive per-parameter 学习率s.

    config format:
    - learning_rate: Scalar 学习率.
    - decay_rate: Scalar between 0 并 1 giving decay rate 用于 squared
      梯度 cache.
    - epsilon: Sm所有 scalar 使用 用于 smoothing 到 avoid dividing by zero.
    - cache: Moving average 的 second moments 的 梯度.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    rho = config["decay_rate"]
    lr = config["learning_rate"]
    eps = config["epsilon"]
    config["cache"] *= rho
    config["cache"] += (1.0 - rho) * dw ** 2
    step = -(lr * dw) / (np.sqrt(config["cache"]) + eps)
    w += step
    next_w = w

    return next_w, config


def adam(w, dw, config=None):
    """
    使用s Adam update rule, which incorporates moving averages 的 both the
    梯度 并 its square 并 a bias correction term.

    config format:
    - learning_rate: Scalar 学习率.
    - beta1: Decay rate 用于 moving average 的 first moment 的 梯度.
    - beta2: Decay rate 用于 moving average 的 second moment 的 梯度.
    - epsilon: Sm所有 scalar 使用 用于 smoothing 到 avoid dividing by zero.
    - m: Moving average 的 梯度.
    - v: Moving average 的 squared 梯度.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]
    t, m, v = config["t"], config["m"], config["v"]
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw * dw)
    t += 1
    alpha = config["learning_rate"] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    w -= alpha * (m / (np.sqrt(v) + eps))
    config["t"] = t
    config["m"] = m
    config["v"] = v
    next_w = w

    return next_w, config
