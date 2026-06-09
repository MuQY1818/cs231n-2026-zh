import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverPytorch(object):
    """
    A CaptioningSolverPytorch encapsulates 所有 logic necessary for
    训练 Pytorch based image captioning 模型.

    To 训练 a 模型, you 将 first construct a CaptioningSolver instance,
    passing 模型, 数据集, 并 various options (学习率, batch size,
    etc) 到 constructor. You 将 然后 调用 训练() method 到 run the
    optimization procedure 并 训练 模型.

    After 训练() method returns, instance 变量 solver.损失_history
    将 contain a list 的 所有 损失 encountered during 训练.

    Example usage 可能 look something like 这个:

    数据 = load_coco_数据()
    模型 = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(模型, 数据,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.训练()


    A CaptioningSolverPytorch works on a 模型 object 该 must conform 到 following
    API:

      输入:
      - 特征: Array giving a minibatch 的 特征 用于 images, 的 形状 (N, D
      - captions: Array 的 captions 用于 those images, 的 形状 (N, T) 其中
        each element is 在 range (0, V].

      返回:
      - 损失: Scalar giving 损失
      - grads: Dictionary 使用 same keys as self.params mapping parameter
        names 到 梯度 的 损失 使用 respect 到 those 参数.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - 模型: A 模型 object conforming 到 API described above
        - 数据: A 字典 的 训练 并 验证 数据 来自 load_coco_数据

        Optional arguments:

        - learning_rate: Learning rate 的 optimizer.
        - batch_size: Size 的 minibatches 使用 到 计算 损失 并 梯度 during
          训练.
        - num_epochs: 数量 epochs 到 run 用于 during 训练.
        - print_every: Integer; 训练 损失 将 be printed every print_every
          iterations.
        - verbose: Boolean; if set 到 false 然后 no 输出 将 be printed during
          训练.
        """
        self.model = model
        self.data = data

        # 解包关键字参数
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(list(model.params.values()), self.learning_rate)

        # 如果存在额外关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping 变量 用于 optimization. Don't 调用 这个
        手动.
        """
        # 设置若干变量用于记录训练过程
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Make a single 梯度 update. This is 调用 by 训练() 并 应该 not
        be 调用 手动.
        """
        # 构造一个训练数据 minibatch
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions = torch.from_numpy(captions).long()
        features = torch.from_numpy(features)
        loss = self.model.loss(features, captions)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.detach().numpy())

    def train(self):
        """
        Run optimization 到 训练 模型.
        """
        for k, v in self.model.params.items():
          v.requires_grad_()

        num_train = self.data["train_captions"].shape[0]
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

            # 每个 epoch 结束时，递增 epoch 计数器。
            epoch_end = (t + 1) % iterations_per_epoch == 0

        for k, v in self.model.params.items():
          v.requires_grad_(False)

