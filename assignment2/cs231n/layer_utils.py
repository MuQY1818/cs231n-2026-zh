from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    一个 convenience layer，执行 affine transform 后接 ReLU。

    输入:
    - x: affine layer 的输入
    - w, b: affine layer 的权重

    返回一个 tuple:
    - out: ReLU 的输出
    - cache: 传给 backward pass 的对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    affine-relu convenience layer 的 backward pass。
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    一个 convenience layer，执行 convolution 后接 ReLU。

    输入:
    - x: convolutional layer 的输入
    - w, b, conv_param: convolutional layer 的权重和参数

    返回一个 tuple:
    - out: ReLU 的输出
    - cache: 传给 backward pass 的对象
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    conv-relu convenience layer 的 backward pass。
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
    一个 convenience layer，执行 convolution、ReLU 和 pool。

    输入:
    - x: convolutional layer 的输入
    - w, b, conv_param: convolutional layer 的权重和参数
    - pool_param: pooling layer 的参数

    返回一个 tuple:
    - out: pooling layer 的输出
    - cache: 传给 backward pass 的对象
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    conv-relu-pool convenience layer 的 backward pass。
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
