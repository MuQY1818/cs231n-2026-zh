from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience 层 该 perorms an affine transform followed by a ReLU

    输入:
    - x: 输入 到 affine 层
    - w, b: Weights 用于 affine 层

    返回 a tuple of:
    - out: Output 来自 ReLU
    - cache: Object 到 give 到 反向传播
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    反向传播 用于 affine-relu convenience 层
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

