import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch


class CaptioningSolver(object):
    """
    CaptioningSolver 封装训练 image captioning 模型所需的所有逻辑。
    CaptioningSolver 使用 optim.py 中定义的不同 update rules 执行
    stochastic gradient descent。

    solver 接收训练和验证数据及标签，因此可以定期检查训练集和验证集上的
    分类 accuracy，以监控 overfitting。

    若要训练模型，首先构造 CaptioningSolver 实例，并将模型、数据集以及
    各种选项（learning rate、batch size 等）传给构造函数。然后调用
    train() 方法运行 optimization procedure 并训练模型。

    train() 方法返回后，model.params 将包含训练过程中在验证集上表现最好的参数。
    此外，实例变量 solver.loss_history 会包含训练期间遇到的所有损失；
    solver.train_acc_history 和 solver.val_acc_history 会分别记录模型在每个
    epoch 的训练集和验证集 accuracy。

    示例用法可能如下：

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    CaptioningSolver 作用于一个 model 对象，该对象必须符合以下 API：

    - model.params 必须是一个字典，将字符串参数名映射到包含参数值的 numpy 数组。

    - model.loss(features, captions) 必须是一个函数，用以下输入和输出计算
      training-time loss 和 gradients：

      输入:
      - features: 数组，给出一个 minibatch 中 images 的 features，形状为 (N, D)
      - captions: 数组，给出这些 images 的 captions，形状为 (N, T)，其中
        每个元素位于区间 (0, V]。

      返回:
      - loss: scalar loss
      - grads: 字典，key 与 self.params 相同，将参数名映射到损失关于这些参数的梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的 CaptioningSolver 实例。

        Required arguments:
        - model: 符合上述 API 的模型对象。
        - data: 来自 load_coco_data 的训练和验证数据字典。

        Optional arguments:
        - update_rule: 字符串，给出 optim.py 中 update rule 的名称。默认值为 'sgd'。
        - optim_config: 字典，包含传给所选 update rule 的 hyperparameters。
          每个 update rule 需要不同 hyperparameters（见 optim.py），但所有
          update rules 都需要 'learning_rate' 参数，因此它应始终存在。
        - lr_decay: 用于 learning rate decay 的 scalar；每个 epoch 后 learning rate
          都会乘以这个值。
        - batch_size: 训练期间用于计算 loss 和 gradients 的 minibatch 大小。
        - num_epochs: 训练运行的 epoch 数量。
        - print_every: 整数；每隔 print_every 次迭代打印一次训练 loss。
        - verbose: Boolean；如果设为 false，训练期间不打印输出。
        """
        self.model = model
        self.data = data

        # 解包关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # 如果存在额外关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # 确认 update rule 存在，然后用实际函数替换字符串名称。
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        设置 optimization 所需的一些 book-keeping 变量。不要手动调用此方法。
        """
        # 设置若干变量用于记录训练过程
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # 为每个参数复制一份 optim_config。
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        执行一次 gradient update。此方法由 train() 调用，不应手动调用。
        """
        # 构造一个训练数据 minibatch
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        # 计算损失和梯度
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        检查模型在给定数据上的 accuracy。

        输入:
        - X: 数据数组，形状为 (N, d_1, ..., d_k)
        - y: labels 数组，形状为 (N,)
        - num_samples: 如果不为 None，则对子采样数据，只在 num_samples 个数据点上测试模型。
        - batch_size: 将 X 和 y 分成此大小的 batches，以避免占用过多内存。

        返回:
        - acc: scalar，表示被模型正确分类的样本比例。
        """
        return 0.0

        # 按需对数据做子采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分 batch 计算预测结果
        num_batches = N / batch_size
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

            # 每个 epoch 结束时，递增 epoch 计数器并衰减 learning rate。
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay
