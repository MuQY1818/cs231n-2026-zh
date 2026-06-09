"""
此文件实现训练 neural networks 常用的各种 first-order update rules。
每个 update rule 接收当前权重以及损失关于这些权重的梯度，并产生下一组权重。
每个 update rule 都具有相同接口：

def update(w, dw, config=None):

输入:
  - w: numpy 数组，表示当前权重。
  - dw: 与 w 形状相同的 numpy 数组，表示损失关于 w 的梯度。
  - config: 字典，包含 learning rate、momentum 等 hyperparameter 值。
    如果 update rule 需要在多次迭代中缓存值，则 config 也会保存这些缓存值。

返回:
  - next_w: update 后的下一个点。
  - config: 传给下一次 update rule 迭代的 config 字典。

注意：对大多数 update rules 来说，默认 learning rate 可能效果不好；
不过其他 hyperparameters 的默认值应能适用于多种不同问题。

出于效率考虑，update rules 可能会执行 in-place updates，直接修改 w，
并令 next_w 等于 w。"""

import numpy as np


def sgd(w, dw, config=None):
    """
    执行 vanilla stochastic gradient descent。

    config format:
    - learning_rate: scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    使用 momentum 执行 stochastic gradient descent。

    config format:
    - learning_rate: scalar learning rate.
    - momentum: 介于 0 和 1 之间的 scalar，表示 momentum 值。
      设置 momentum = 0 会退化为 sgd。
    - velocity: 与 w 和 dw 形状相同的 numpy 数组，用于存储梯度的 moving average。
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
    使用 RMSProp update rule。它使用 squared gradient 值的 moving average
    来设置每个参数的自适应 learning rate。

    config format:
    - learning_rate: scalar learning rate.
    - decay_rate: 介于 0 和 1 之间的 scalar，表示 squared gradient cache 的 decay rate。
    - epsilon: 用于 smoothing 的小 scalar，以避免除以零。
    - cache: gradient second moments 的 moving average。
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
    使用 Adam update rule。它结合 gradient 及其平方的 moving averages，
    并包含 bias correction 项。

    config format:
    - learning_rate: scalar learning rate.
    - beta1: gradient first moment moving average 的 decay rate。
    - beta2: gradient second moment moving average 的 decay rate。
    - epsilon: 用于 smoothing 的小 scalar，以避免除以零。
    - m: gradient 的 moving average。
    - v: squared gradient 的 moving average。
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
