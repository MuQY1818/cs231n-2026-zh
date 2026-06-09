import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverPytorch(object):
    """
    CaptioningSolverPytorch 封装了训练基于 PyTorch 的 image captioning 模型所需的
    全部逻辑。

    要训练模型，首先构造一个 CaptioningSolver 实例，将模型、数据集以及各种选项
    （learning rate、batch size 等）传给构造函数。然后调用 train() 方法运行
    optimization 过程并训练模型。

    train() 方法返回后，实例变量 solver.loss_history 会包含训练过程中遇到的所有
    loss。

    示例用法大致如下:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    CaptioningSolverPytorch 作用于一个必须符合以下 API 的 model 对象:

      输入:
      - features: 给出图像特征 minibatch 的数组，形状为 (N, D)
      - captions: 这些图像对应的 captions 数组，形状为 (N, T)，其中每个元素都在
        (0, V] 范围内。

      返回:
      - loss: 标量 loss
      - grads: 字典，键与 self.params 相同，将参数名映射到 loss 关于这些参数的梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的 CaptioningSolver 实例。

        Required arguments:
        - model: 符合上述 API 的 model 对象
        - data: 来自 load_coco_data 的训练和验证数据字典

        Optional arguments:

        - learning_rate: optimizer 的 learning rate。
        - batch_size: 训练时用于计算 loss 和梯度的 minibatch 大小。
        - num_epochs: 训练期间运行的 epoch 数。
        - print_every: 整数；每隔 print_every 次 iteration 打印 training loss。
        - verbose: Boolean；如果设为 false，则训练期间不打印输出。
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
        设置 optimization 需要的记录变量。不要手动调用这个方法。
        """
        # 设置若干变量用于记录训练过程
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        执行单次梯度更新。该方法由 train() 调用，不应手动调用。
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
        运行 optimization 来训练模型。
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
