from .layers import *
from .fast_layers import *


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


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience 层 该 performs a convolution followed by a ReLU.

    输入:
    - x: 输入 到 convolutional 层
    - w, b, conv_param: Weights 并 参数 用于 convolutional 层

    返回 a tuple of:
    - out: Output 来自 ReLU
    - cache: Object 到 give 到 反向传播
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    反向传播 用于 conv-relu convenience 层.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience 层 该 performs a convolution, a ReLU, 并 a pool.

    输入:
    - x: 输入 到 convolutional 层
    - w, b, conv_param: Weights 并 参数 用于 convolutional 层
    - pool_param: Parameters 用于 pooling 层

    返回 a tuple of:
    - out: Output 来自 pooling 层
    - cache: Object 到 give 到 反向传播
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    反向传播 用于 conv-relu-pool convenience 层
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
