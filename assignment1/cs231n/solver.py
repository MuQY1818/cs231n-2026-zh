from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """
    A Solver encapsulates 所有 logic necessary 用于训练 分类
    模型. Solver performs stochastic 梯度 descent 使用 different
    update rules defined 在 optim.py.

    solver accepts both 训练 并 验证 数据 并 标签 so it 可以
    periodi调用y check 分类 accuracy on both 训练 并 验证
    数据 到 watch out 用于 过拟合ting.

    To 训练 a 模型, you 将 first construct a Solver instance, passing the
    模型, 数据集, 并 various options (学习率, batch size, etc) 到 the
    constructor. You 将 然后 调用 训练() method 到 run optimization
    procedure 并 训练 模型.

    After 训练() method returns, 模型.params 将 contain 参数
    该 performed best on 验证 set 在 course 的 训练.
    In addition, instance 变量 solver.损失_history 将 contain a list
    of 所有 损失 encountered during 训练 并 instance 变量
    solver.训练_acc_history 并 solver.val_acc_history 将 be lists 的 the
    accuracies 模型的 on 训练 并 验证 set at each epoch.

    Example usage 可能 look something like 这个:

    数据 = {
      'X_训练': # 训练数据
      'y_训练': # 训练 标签
      'X_val': # 验证 数据
      'y_val': # 验证 标签
    }
    模型 = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(模型, 数据,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.训练()


    A Solver works on a 模型 object 该 must conform 到 following API:

    - 模型.params must be a 字典 mapping string parameter names 到 numpy
      数组 containing parameter 值.

    - 模型.损失(X, y) must be a 函数 该 计算 训练时 损失 and
      梯度, 并 测试时 分类 分数, 使用 following 输入
      and 输出:

      输入:
      - X: Array giving a minibatch 的 输入 数据 的 形状 (N, d_1, ..., d_k)
      - y: Array 的 标签, 的 形状 (N,) giving 标签 用于 X 其中 y[i] is the
        标签 用于 X[i].

      返回:
      If y is None, run a 测试时 前向传播 并 return:
      - 分数: Array 的 形状 (N, C) giving 分类 分数 用于 X 其中
        分数[i, c] gives score 的 类别 c 用于 X[i].

      If y is not None, run a 训练 time 前向 并 反向传播 and
      return a tuple of:
      - 损失: Scalar giving 损失
      - grads: Dictionary 使用 same keys as self.params mapping parameter
        names 到 梯度 的 损失 使用 respect 到 those 参数.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - 模型: A 模型 object conforming 到 API described above
        - 数据: A 字典 的 训练 并 验证 数据 containing:
          'X_训练': Array, 形状 (N_训练, d_1, ..., d_k) 的 训练 images
          'X_val': Array, 形状 (N_val, d_1, ..., d_k) 的 验证 images
          'y_训练': Array, 形状 (N_训练,) 的 标签 用于训练 images
          'y_val': Array, 形状 (N_val,) 的 标签 用于 验证 images

        Optional arguments:
        - update_rule: A string giving name 的 an update rule 在 optim.py.
          Default is 'sgd'.
        - optim_config: A 字典 containing hyper参数 该 将 be
          passed 到 chosen update rule. Each update rule requires different
          hyper参数 (see optim.py) but 所有 update rules require a
          'learning_rate' parameter so 该 应该 always be present.
        - lr_decay: A scalar 用于 学习率 decay; after each epoch the
          学习率 is multiplied by 这个 值.
        - batch_size: Size 的 minibatches 使用 到 计算 损失 并 梯度
          during 训练.
        - num_epochs: 数量 epochs 到 run 用于 during 训练.
        - print_every: Integer; 训练 损失 将 be printed every
          print_every iterations.
        - verbose: Boolean; if set 到 false 然后 no 输出 将 be printed
          during 训练.
        - num_训练_样本: 数量 训练 样本 使用 到 check 训练
          accuracy; default is 1000; set 到 None 到 使用 entire 训练集.
        - num_val_样本: 数量 验证 样本 到 使用 到 check val
          accuracy; default is None, which 使用s entire 验证 set.
        - check点_name: If not None, 然后 save 模型 check点 here every
          epoch.
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # 解包关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # 如果存在额外关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure update rule exists, 然后 replace string
        # name 使用 actual 函数
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping 变量 用于 optimization. Don't 调用 这个
        手动.
        """
        # 设置若干变量用于记录训练过程
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy 的 optim_config 用于 each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single 梯度 update. This is 调用 by 训练() 并 应该 not
        be 调用 手动.
        """
        # 构造一个训练数据 minibatch
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy 模型的 on provided 数据.

        输入:
        - X: Array 的 数据, 的 形状 (N, d_1, ..., d_k)
        - y: Array 的 标签, 的 形状 (N,)
        - num_样本: If not None, subsample 数据 并 only 测试 模型
          on num_样本 数据点.
        - batch_size: Split X 并 y 到 batches 的 这个 size 到 avoid 使用
          too much memory.

        返回:
        - acc: Scalar giving fraction 的 instances 该 were correctly
          类别ified by 模型.
        """

        # 按需对数据做子采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分 batch 计算预测结果
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization 到 训练 模型.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # 按需打印训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At end 的 every epoch, increment epoch counter 并 decay
            # 学习率.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check 训练 并 val accuracy on first iteration, last
            # iteration, 并 at end 的 each epoch.
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # 跟踪最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # 训练结束时，把最佳参数写回模型
        self.model.params = self.best_params
