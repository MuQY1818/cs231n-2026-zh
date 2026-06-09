from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive 实现 的 numerical 梯度 的 f at x
    - f 应为 a 函数 该 takes a single argument
    - x is 点 (numpy 数组) 到 evaluate 梯度 at
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
    Evaluate a numeric 梯度 用于 a 函数 该 accepts a numpy
    数组 并 returns a numpy 数组.
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
    计算 numeric 梯度 用于 a 函数 该 operates on 输入
    and 输出 blobs.

    We assume 该 f accepts several 输入 blobs as arguments, followed by a
    blob 其中 输出 将 be written. For 样本, f 可能 be 调用 like:

    f(x, w, out)

    其中 x 并 w are 输入 Blobs, 并 结果 的 f 将 be written 到 out.

    输入:
    - f: 函数
    - 输入: tuple 的 输入 blobs
    - 输出: 输出 blob
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
    sample a few random elements 并 only return numerical
    in 这个 维度.
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
