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
    执行 vanilla stochastic gradient descent。

    config format:
    - learning_rate: 标量 learning rate。
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
    - learning_rate: 标量 learning rate。
    - momentum: 0 到 1 之间的标量，给出 momentum 值。
      设置 momentum = 0 时退化为 sgd。
    - velocity: 与 w 和 dw 形状相同的 numpy 数组，用于存储梯度的 moving average。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO：实现 momentum update formula。将更新后的值存入 next_w 变量。还应    #
    # 使用并更新 velocity v。                                                   #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用 RMSProp update rule。该规则使用 squared gradient 值的 moving average，
    为每个参数设置自适应 learning rate。

    config format:
    - learning_rate: 标量 learning rate。
    - decay_rate: 0 到 1 之间的标量，给出 squared gradient cache 的 decay rate。
    - epsilon: 用于 smoothing 的小标量，避免除以 0。
    - cache: 梯度二阶矩的 moving average。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO：实现 RMSprop update formula，将 w 的下一个值存入 next_w 变量。      #
    # 不要忘记更新存储在 config['cache'] 中的 cache 值。                       #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用 Adam update rule。该规则结合了梯度及其平方的 moving average，以及
    bias correction 项。

    config format:
    - learning_rate: 标量 learning rate。
    - beta1: 梯度一阶矩 moving average 的 decay rate。
    - beta2: 梯度二阶矩 moving average 的 decay rate。
    - epsilon: 用于 smoothing 的小标量，避免除以 0。
    - m: 梯度的 moving average。
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
    ###########################################################################
    # TODO：实现 Adam update formula，将 w 的下一个值存入 next_w 变量。不要忘记 #
    # 更新 config 中存储的 m、v 和 t 变量。                                    #
    #                                                                         #
    # 注意：为了匹配参考输出，请在任何计算使用 t 之前先修改 t。                 #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                            #
    ###########################################################################

    return next_w, config
