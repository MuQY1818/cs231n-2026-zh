import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


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
    ###########################################################################
    # TODO：实现 momentum update formula. 将更新后的 值 在 #
    # next_w 变量. 你应该 also 使用 并 update velocity v.     #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
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
    ###########################################################################
    # TODO：实现 RMSprop update formula, 将下一个值存储 的 w #
    # in next_w 变量. 不要忘记 到 update cache 值 存储 在    #
    # config['cache'].                                                        #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

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
    ###########################################################################
    # TODO：实现 Adam update formula, 将下一个值存储 的 w 在 #
    # next_w 变量. 不要忘记 到 update m, v, 并 t 变量   #
    # 存储 在 config.                                                       #
    #                                                                         #
    # 注意： In order 到 match 参考 输出, please modify t _before_  #
    # 使用 it 在 any calculations.                                           #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return next_w, config
