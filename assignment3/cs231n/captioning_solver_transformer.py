import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverTransformer(object):
    """
    CaptioningSolverTransformer 封装训练基于 Transformer 的 image captioning
    模型所需的所有逻辑。

    若要训练模型，首先构造 CaptioningSolver 实例，并将模型、数据集以及
    各种选项（learning rate、batch size 等）传给构造函数。然后调用
    train() 方法运行 optimization procedure 并训练模型。

    train() 方法返回后，实例变量 solver.loss_history 会包含训练期间遇到的所有损失。

    示例用法可能如下：

    data = load_coco_data()
    model = MyAwesomeTransformerModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    CaptioningSolverTransformer 作用于一个 model 对象，该对象必须符合以下 API：

      输入:
      - features: 数组，给出一个 minibatch 中 images 的 features，形状为 (N, D)
      - captions: 数组，给出这些 images 的 captions，形状为 (N, T)，其中
        每个元素位于区间 (0, V]。

      返回:
      - loss: scalar loss
      - grads: 字典，key 与 self.params 相同，将参数名映射到损失关于这些参数的梯度。
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        构造一个新的 CaptioningSolver 实例。

        Required arguments:
        - model: 符合上述 API 的模型对象。
        - data: 来自 load_coco_data 的训练和验证数据字典。

        Optional arguments:

        - learning_rate: optimizer 的 learning rate。
        - batch_size: 训练期间用于计算 loss 和 gradients 的 minibatch 大小。
        - num_epochs: 训练运行的 epoch 数量。
        - print_every: 整数；每隔 print_every 次迭代打印一次训练 loss。
        - verbose: Boolean；如果设为 false，训练期间不打印输出。
        """
        self.model = model
        self.data = data

        # 解包关键字参数
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # 如果存在额外关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

        self.idx_to_word = idx_to_word

    def _reset(self):
        """
        设置 optimization 所需的一些 book-keeping 变量。不要手动调用此方法。
        """
        # 设置若干变量用于记录训练过程
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        执行一次 gradient update。此方法由 train() 调用，不应手动调用。
        """
        # 构造一个训练数据 minibatch
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self.model._null

        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)

        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        运行 optimization 来训练模型。
        """
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

    def transformer_temporal_softmax_loss(self, x, y, mask):
        """
        用于 RNNs 的 temporal softmax loss 版本。假设我们在大小为 V 的词表上，
        对长度为 T 的 timeseries 的每个时间步进行预测，minibatch 大小为 N。
        输入 x 给出所有时间步上所有 vocabulary elements 的分数，y 给出每个
        时间步的 ground-truth element 索引。我们在每个时间步使用 cross-entropy
        loss，并对所有时间步的 loss 求和，再在 minibatch 上取平均。

        额外的复杂之处在于，我们可能希望忽略某些时间步上的模型输出，因为不同长度的
        sequences 可能被合并进同一个 minibatch，并用 NULL tokens padding。
        可选的 mask 参数会告诉我们哪些元素应参与 loss 计算。

        输入:
        - x: 输入分数，形状为 (N, T, V)
        - y: Ground-truth indices，形状为 (N, T)，其中每个元素在范围
             0 <= y[i, t] < V
        - mask: Boolean 数组，形状为 (N, T)，其中 mask[i, t] 表示
          x[i, t] 处的分数是否应参与 loss 计算。

        返回:
        - loss: scalar loss
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
