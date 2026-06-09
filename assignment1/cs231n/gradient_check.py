from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    在 x 处计算 f 的数值梯度的朴素实现。
    - f 应该是一个接收单个参数的函数。
    - x 是要计算梯度的点（numpy 数组）。
    """

    fx = f(x)  # 在原始点计算函数值
    grad = np.zeros_like(x)
    # 遍历 x 中的所有索引
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # 在 x+h 处计算函数值
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # 计算 f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # 计算 f(x - h)
        x[ix] = oldval  # 恢复原值

        # 使用中心差分公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)  # 斜率
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # 前进到下一个维度

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    为一个接收 numpy 数组并返回 numpy 数组的函数计算数值梯度。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    为一个作用于输入和输出 blobs 的函数计算数值梯度。

    假设 f 接收若干输入 blobs 作为参数，最后跟一个用于写入输出的 blob。
    例如，f 可能这样调用：

    f(x, w, out)

    其中 x 和 w 是输入 blobs，f 的结果会写入 out。

    输入:
    - f: 函数
    - inputs: 输入 blobs 组成的 tuple
    - output: 输出 blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    随机采样若干元素，只返回这些维度上的数值梯度。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # 增加 h
        fxmh = f(x)  # 计算 f(x - h)
        x[ix] = oldval  # 重置

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
