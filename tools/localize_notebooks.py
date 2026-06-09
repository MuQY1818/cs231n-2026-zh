from pathlib import Path

import nbformat
from nbformat.validator import normalize


ROOT = Path(__file__).resolve().parents[1]


AI_USE = """**生成式 AI 使用说明**：本作业中使用生成式 AI 工具时，适用与协作相同的课程政策。和其他协作者一样，每位学生都必须独立写出自己的解答，不能直接依赖交互输出；提交内容中还应注明协作的性质。使用生成式 AI 工具实质性完成作业内容不符合本作业的精神，也会违反 [Honor Code](https://communitystandards.stanford.edu/policies-and-guidance/honor-code)。"""


STUDENT_DECLARATION = """**学生声明（必须填写）**

**SUNet id：**  
_填写你的 SUNet id_

**你是否使用了生成式 AI 工具（例如 ChatGPT、Copilot 等）？**  
- [ ] 否  
- [ ] 是（请在下方说明）

**如果使用了，请说明你如何使用生成式 AI，以及用于作业的哪些部分：**  
_填写你的回答_

**请确认所有提交内容都反映你自己的理解，并且你没有依赖 AI 完成作业的实质性部分：**  
- [ ] 我确认"""


TRANSLATIONS = {
    ("assignment1/FullyConnectedNets.ipynb", 0): AI_USE,
    ("assignment1/FullyConnectedNets.ipynb", 2): """# 多层全连接网络
在这个练习中，你将实现一个可以包含任意数量隐藏层的全连接网络。""",
    ("assignment1/FullyConnectedNets.ipynb", 4): """阅读 `cs231n/classifiers/fc_net.py` 文件中的 `FullyConnectedNet` 类。

实现网络初始化、前向传播和反向传播。在整个作业中，你会在 `cs231n/layers.py` 中实现不同层。这里可以复用你之前实现的 `affine_forward`、`affine_backward`、`relu_forward`、`relu_backward` 和 `softmax_loss`。目前还不需要实现 dropout 或 batch/layer normalization；这些功能会在后续部分加入。""",
    ("assignment1/FullyConnectedNets.ipynb", 7): """## 初始损失与梯度检查

作为合理性检查，运行下面的代码来检查初始损失，并分别在有正则化和无正则化的情况下对网络做梯度检查。这有助于判断初始损失是否在合理范围内。

做梯度检查时，你应该看到大约 `1e-7` 或更小的误差。""",
    ("assignment1/FullyConnectedNets.ipynb", 9): """另一个合理性检查是确认网络能在一个只有 50 张图片的小数据集上过拟合。首先尝试一个三层网络，每个隐藏层有 100 个单元。在下面的单元中，调整**学习率**和**权重初始化尺度**，让模型在 20 个 epoch 内过拟合并达到 100% 的训练准确率。""",
    ("assignment1/FullyConnectedNets.ipynb", 11): """现在尝试使用一个五层网络，每层有 100 个单元，在 50 个训练样本上过拟合。同样，你需要调整学习率和权重初始化尺度，但应该能够在 20 个 epoch 内达到 100% 的训练准确率。""",
    ("assignment1/FullyConnectedNets.ipynb", 13): """## 内联问题 1：
你是否注意到三层网络和五层网络在训练难度上的差异？特别是根据你的实验经验，哪个网络似乎对初始化尺度更敏感？你认为为什么会这样？

## 回答：
[在此填写]""",
    ("assignment1/FullyConnectedNets.ipynb", 14): """# 更新规则
到目前为止，我们一直使用普通的随机梯度下降（SGD）作为更新规则。更复杂的更新规则可以让深度网络更容易训练。接下来我们将实现几种最常用的更新规则，并将它们与普通 SGD 进行比较。""",
    ("assignment1/FullyConnectedNets.ipynb", 15): """## SGD+Momentum
带动量的随机梯度下降是一种广泛使用的更新规则，通常能让深度网络比普通随机梯度下降收敛得更快。更多信息请参考 http://cs231n.github.io/neural-networks-3/#sgd 中的 Momentum Update 部分。

打开 `cs231n/optim.py` 文件，阅读文件开头的文档，确保你理解 API。然后在 `sgd_momentum` 函数中实现 SGD+momentum 更新规则，并运行下面的代码检查你的实现。你应该看到小于 `1e-8` 的误差。""",
    ("assignment1/FullyConnectedNets.ipynb", 17): """完成后，运行下面的代码，分别用 SGD 和 SGD+momentum 训练一个六层网络。你应该会看到 SGD+momentum 收敛得更快。""",
    ("assignment1/FullyConnectedNets.ipynb", 19): """## RMSProp 和 Adam
RMSProp [1] 和 Adam [2] 是两种更新规则，它们通过梯度二阶矩的滑动平均为每个参数设置不同的学习率。

在 `cs231n/optim.py` 文件中，在 `rmsprop` 函数里实现 RMSProp 更新规则，在 `adam` 函数里实现 Adam 更新规则，并使用下面的测试检查实现。

**注意：** 请实现_完整的_ Adam 更新规则（包含偏差校正机制），不要实现课程笔记中最先提到的简化版本。

[1] Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4 (2012).

[2] Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.""",
    ("assignment1/FullyConnectedNets.ipynb", 22): """调试好 RMSProp 和 Adam 的实现后，运行下面的代码，用这些新的更新规则训练一组深层网络：""",
    ("assignment1/FullyConnectedNets.ipynb", 24): """## 内联问题 2：

AdaGrad 和 Adam 一样，也是按参数自适应的优化方法，它使用如下更新规则：

```
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```

John 注意到，当他用 AdaGrad 训练网络时，更新量会变得非常小，网络学习得很慢。根据你对 AdaGrad 更新规则的理解，为什么更新量会变得很小？Adam 会有同样的问题吗？

## 回答：
[在此填写]""",
    ("assignment1/FullyConnectedNets.ipynb", 25): """# 训练一个好模型
在 CIFAR-10 上训练你能得到的最好的全连接模型，并把最佳模型存入 `best_model` 变量。我们要求使用全连接网络在验证集上至少达到 50% 的准确率。

如果调参足够仔细，达到 55% 以上是可能的；但这一部分不强制要求，也不会因此给额外分。下一次作业会要求你在 CIFAR-10 上训练尽可能好的卷积网络，我们更希望你把精力投入到卷积网络上，而不是继续深挖全连接网络。

**注意：** 在下一次作业中，你会学习 BatchNormalization 和 Dropout 等技术，它们可以帮助训练更强的模型。""",
    ("assignment1/FullyConnectedNets.ipynb", 27): """# 测试你的模型
在验证集和测试集上运行你的最佳模型。你应该在验证集和测试集上都达到至少 50% 的准确率。""",

    ("assignment1/collect_submission.ipynb", 0): AI_USE,
    ("assignment1/collect_submission.ipynb", 2): """# 收集提交文件：压缩代码并生成 PDF

完成所有其他 notebook 后再运行这个 notebook：`knn.ipynb`、`softmax.ipynb`、`two_layer_net.ipynb`、`features.ipynb` 和 `FullyConnectedNets.ipynb`。

它会：

* 生成一个包含你的代码（`.py` 和 `.ipynb`）的压缩包，文件名为 `a1_code_submission.zip`。
* 将所有 notebook 转换成一个 PDF 文件，文件名为 `a1_inline_submission.pdf`。

如果这一步提交文件生成成功，你应该会看到如下显示信息：

`### Done! Please submit a1_code_submission.zip and a1_inline_submission.pdf to Gradescope. ###`

请确保把 zip 和 pdf 文件下载到本地电脑，然后提交到 Gradescope。恭喜你完成本次作业。""",

    ("assignment1/features.ipynb", 0): AI_USE,
    ("assignment1/features.ipynb", 2): """# 图像特征练习
*请完成并提交这份 worksheet，包括其中的输出以及 worksheet 外部的任何辅助代码。更多细节请参考课程网站上的 [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html)。*

我们已经看到，在图像分类任务中，直接在输入图像的像素上训练线性分类器可以达到还不错的性能。在这个练习中，我们将展示：如果不是在原始像素上训练线性分类器，而是在从原始像素计算出的特征上训练分类器，分类性能可以进一步提升。

本练习的所有工作都将在这个 notebook 中完成。""",
    ("assignment1/features.ipynb", 4): """## 加载数据
和前面的练习类似，我们将从磁盘加载 CIFAR-10 数据。""",
    ("assignment1/features.ipynb", 6): """## 提取特征
对每张图像，我们会计算方向梯度直方图（Histogram of Oriented Gradients，HOG），并使用 HSV 颜色空间中的 hue 通道计算颜色直方图。最终，我们通过拼接 HOG 特征向量和颜色直方图特征向量，为每张图像构造特征向量。

粗略来说，HOG 应该能捕捉图像纹理，同时忽略颜色信息；颜色直方图表示输入图像的颜色，同时忽略纹理。因此，我们预期同时使用二者会比只使用其中一种效果更好。你也可以出于兴趣自己验证这个假设。

`hog_feature` 和 `color_histogram_hsv` 函数都作用于单张图像，并返回该图像的特征向量。`extract_features` 函数接收一组图像和一组特征函数，对每张图像运行每个特征函数，并将结果存储在一个矩阵中；矩阵的每一列都是某一张图像所有特征向量拼接后的结果。""",
    ("assignment1/features.ipynb", 8): """## 在特征上训练 Softmax 分类器
使用本作业前面实现的 Softmax 代码，在上面提取的特征上训练 Softmax 分类器；这应该比直接在原始像素上训练得到更好的结果。""",
    ("assignment1/features.ipynb", 13): """### 内联问题 1：
描述你看到的误分类结果。它们是否合理？


$\\color{blue}{\\textit 你的回答：}$""",
    ("assignment1/features.ipynb", 14): """## 在图像特征上训练神经网络
在本作业前面，我们看到在原始像素上训练两层神经网络，比在原始像素上训练线性分类器有更好的分类性能。在这个 notebook 中，我们又看到在图像特征上训练线性分类器，优于在原始像素上训练线性分类器。

为了完整起见，我们还应该尝试在图像特征上训练神经网络。这种方法应该优于前面所有方法：你应该很容易在测试集上达到 55% 以上的分类准确率；我们的最佳模型大约达到 60% 的分类准确率。""",

    ("assignment1/knn.ipynb", 0): AI_USE,
    ("assignment1/knn.ipynb", 2): """# k-Nearest Neighbor（kNN）练习

*请完成并提交这份 worksheet，包括其中的输出以及 worksheet 外部的任何辅助代码。更多细节请参考课程网站上的 [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html)。*

kNN 分类器包含两个阶段：

- 训练阶段：分类器接收训练数据，并简单地记住它们。
- 测试阶段：kNN 将每张测试图像与所有训练图像进行比较，并把最相似的 k 个训练样本的标签转移过来进行分类。
- k 的取值通过交叉验证确定。

在这个练习中，你将实现这些步骤，并理解图像分类的基本流程、交叉验证，同时熟练编写高效的向量化代码。""",
    ("assignment1/knn.ipynb", 8): """现在我们希望使用 kNN 分类器对测试数据进行分类。回忆一下，这个过程可以分成两步：

1. 首先计算所有测试样本和所有训练样本之间的距离。
2. 给定这些距离后，对每个测试样本找到 k 个最近的训练样本，并让它们对标签投票。

先从计算所有训练样本和测试样本之间的距离矩阵开始。例如，如果有 **Ntr** 个训练样本和 **Nte** 个测试样本，这一步应该得到一个 **Nte x Ntr** 的矩阵，其中元素 `(i, j)` 表示第 i 个测试样本和第 j 个训练样本之间的距离。

**注意：本 notebook 要求你实现三种距离计算方式，在这些实现中不能使用 numpy 提供的 `np.linalg.norm()` 函数。**

首先打开 `cs231n/classifiers/k_nearest_neighbor.py`，实现 `compute_distances_two_loops` 函数。这个函数使用一个非常低效的双重循环，遍历所有（测试样本、训练样本）对，并逐元素计算距离矩阵。""",
    ("assignment1/knn.ipynb", 11): """**内联问题 1**

注意距离矩阵中的结构化模式：有些行或列明显更亮。（默认配色中，黑色表示距离小，白色表示距离大。）

- 数据中的什么原因会导致明显更亮的行？
- 什么原因会导致更亮的列？

$\\color{blue}{\\textit 你的回答：}$ *在此填写。*""",
    ("assignment1/knn.ipynb", 13): """你应该看到大约 `27%` 的准确率。现在尝试更大的 `k`，例如 `k = 5`：""",
    ("assignment1/knn.ipynb", 15): """你应该会看到它比 `k = 1` 的表现略好。""",
    ("assignment1/knn.ipynb", 16): """**内联问题 2**

我们也可以使用其他距离度量，例如 L1 距离。对于某张图像 $I_k$ 在位置 $(i,j)$ 的像素值 $p_{ij}^{(k)}$，

所有图像所有像素上的均值 $\\mu$ 为：
$$\\mu=\\frac{1}{nhw}\\sum_{k=1}^n\\sum_{i=1}^{h}\\sum_{j=1}^{w}p_{ij}^{(k)}$$
所有图像在每个像素位置上的逐像素均值 $\\mu_{ij}$ 为：
$$\\mu_{ij}=\\frac{1}{n}\\sum_{k=1}^np_{ij}^{(k)}.$$
整体标准差 $\\sigma$ 和逐像素标准差 $\\sigma_{ij}$ 也以类似方式定义。

如果最近邻分类器使用 L1 距离，下列哪些预处理步骤不会改变它的性能？请选择所有适用项。为避免歧义，训练样本和测试样本都会以相同方式预处理。

1. 减去均值 $\\mu$（$\\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\\mu$）。
2. 减去逐像素均值 $\\mu_{ij}$（$\\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\\mu_{ij}$）。
3. 减去均值 $\\mu$，再除以标准差 $\\sigma$。
4. 减去逐像素均值 $\\mu_{ij}$，再除以逐像素标准差 $\\sigma_{ij}$。
5. 旋转数据的坐标轴，也就是把所有图像旋转相同角度。旋转造成的空白区域用相同像素值填充，不做插值。

$\\color{blue}{\\textit 你的回答：}$


$\\color{blue}{\\textit 你的解释：}$""",
    ("assignment1/knn.ipynb", 20): """### 交叉验证

我们已经实现了 k-Nearest Neighbor 分类器，但之前随意设置了 `k = 5`。现在我们将使用交叉验证来确定这个超参数的最佳取值。""",
    ("assignment1/knn.ipynb", 24): """**内联问题 3**

在分类设置中，关于 $k$-Nearest Neighbor（$k$-NN）的下列说法哪些对所有 $k$ 都成立？请选择所有适用项。
1. k-NN 分类器的决策边界是线性的。
2. 1-NN 的训练误差总是小于或等于 5-NN 的训练误差。
3. 1-NN 的测试误差总是小于 5-NN 的测试误差。
4. 使用 k-NN 分类器对一个测试样本分类所需的时间，会随着训练集大小增长而增长。
5. 以上都不对。

$\\color{blue}{\\textit 你的回答：}$


$\\color{blue}{\\textit 你的解释：}$""",

    ("assignment1/softmax.ipynb", 0): AI_USE,
    ("assignment1/softmax.ipynb", 2): """# Softmax 分类器练习

*请完成并提交这份 worksheet，包括其中的输出以及 worksheet 外部的任何辅助代码。更多细节请参考课程网站上的 [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html)。*

在这个练习中，你将：

- 为 Softmax 分类器实现完全向量化的**损失函数**。
- 实现其**解析梯度**的完全向量化表达式。
- 使用数值梯度**检查你的实现**。
- 使用验证集**调优学习率和正则化**强度。
- 使用 **SGD** **优化**损失函数。
- **可视化**最终学到的权重。""",
    ("assignment1/softmax.ipynb", 4): """## CIFAR-10 数据加载与预处理""",
    ("assignment1/softmax.ipynb", 10): """## Softmax 分类器

本节的代码都将写在 `cs231n/classifiers/softmax.py` 中。

你可以看到，我们已经预填了 `softmax_loss_naive` 函数，它使用 for 循环来计算 softmax 损失函数。""",
    ("assignment1/softmax.ipynb", 12): """**内联问题 1**

为什么我们预期损失会接近 `-log(0.1)`？请简要解释。

$\\color{blue}{\\textit 你的回答：}$ *在此填写*""",
    ("assignment1/softmax.ipynb", 13): """上面函数返回的 `grad` 现在全为零。请推导并实现 softmax 损失函数的梯度，并直接在 `softmax_loss_naive` 函数内部实现。把新代码穿插到现有函数中会比较方便。

为了检查梯度是否实现正确，你可以用数值方法估计损失函数的梯度，并与你计算出的解析梯度进行比较。我们已经提供了完成这件事的代码：""",
    ("assignment1/softmax.ipynb", 15): """**内联问题 2**

虽然 softmax 损失的梯度检查通常很可靠，但对 SVM 损失来说，偶尔会有某个维度在梯度检查中无法完全匹配。这种差异可能由什么造成？这是否值得担心？请给出一个一维的简单例子，说明 SVM 损失的梯度检查可能失败。改变 margin 会如何影响这种情况发生的频率？

注意，样本 $(x_i, y_i)$ 的 SVM 损失定义为：
$$L_i = \\sum_{j\\ne y_i}\\max(0, s_j - s_{y_i} + \\Delta)$$
其中 $j$ 遍历除正确类别 $y_i$ 之外的所有类别，$s_j$ 表示第 $j$ 类的分类器分数，$\\Delta$ 是标量 margin。更多信息请参考 [this](https://cs231n.github.io/linear-classify/) 页面中的 “Multiclass Support Vector Machine loss”。

*提示：严格来说，SVM 损失函数并不是处处可微。*

$\\color{blue}{\\textit 你的回答：}$ *在此填写。*""",
    ("assignment1/softmax.ipynb", 18): """### 随机梯度下降

现在我们已经有了损失和梯度的向量化高效表达式，并且梯度与数值梯度匹配。因此，可以开始用 SGD 最小化损失。本部分代码将写在 `cs231n/classifiers/linear_classifier.py` 中。""",
    ("assignment1/softmax.ipynb", 28): """**内联问题 3**

描述你可视化出来的 Softmax 分类器权重看起来是什么样子，并简要解释为什么会呈现这种形态。

$\\color{blue}{\\textit 你的回答：}$ *在此填写*""",
    ("assignment1/softmax.ipynb", 29): """**内联问题 4** - *True or False*

假设总训练损失定义为所有训练样本逐样本损失的总和。是否可能向训练集中加入一个新样本，使 softmax 损失发生变化，但 SVM 损失保持不变？

$\\color{blue}{\\textit 你的回答：}$


$\\color{blue}{\\textit 你的解释：}$""",

    ("assignment1/two_layer_net.ipynb", 0): AI_USE,
    ("assignment1/two_layer_net.ipynb", 2): """# 全连接神经网络
在这个练习中，我们将用模块化方式实现全连接网络。对每一层，我们都会实现一个 `forward` 函数和一个 `backward` 函数。`forward` 函数接收输入、权重和其他参数，并返回输出以及一个 `cache` 对象；`cache` 保存反向传播所需的数据，例如：

```python
def layer_forward(x, w):
  \"\"\"接收输入 x 和权重 w\"\"\"
  # 做一些计算 ...
  z = # ... 某个中间值
  # 再做一些计算 ...
  out = # 输出

  cache = (x, w, z, out) # 计算梯度时需要的值

  return out, cache
```

反向传播会接收上游导数和 `cache` 对象，并返回关于输入和权重的梯度，例如：

```python
def layer_backward(dout, cache):
  \"\"\"
  接收 dout（损失对输出的导数）和 cache，
  并计算对输入的导数。
  \"\"\"
  # 解包 cache 中的值
  x, w, z, out = cache

  # 使用 cache 中的值计算导数
  dx = # 损失对 x 的导数
  dw = # 损失对 w 的导数

  return dx, dw
```

用这种方式实现多个层之后，我们就能很容易地组合它们，构建不同架构的分类器。""",
    ("assignment1/two_layer_net.ipynb", 5): """# Affine 层：前向传播
打开 `cs231n/layers.py` 文件，实现 `affine_forward` 函数。

完成后，可以运行下面的代码测试你的实现：""",
    ("assignment1/two_layer_net.ipynb", 7): """# Affine 层：反向传播
现在实现 `affine_backward` 函数，并使用数值梯度检查来测试你的实现。""",
    ("assignment1/two_layer_net.ipynb", 9): """# ReLU 激活：前向传播
在 `relu_forward` 函数中实现 ReLU 激活函数的前向传播，并使用下面的代码测试你的实现：""",
    ("assignment1/two_layer_net.ipynb", 11): """# ReLU 激活：反向传播
现在在 `relu_backward` 函数中实现 ReLU 激活函数的反向传播，并使用数值梯度检查测试你的实现：""",
    ("assignment1/two_layer_net.ipynb", 13): """## 内联问题 1：

我们只要求你实现 ReLU，但神经网络中还有许多不同的激活函数，每种都有优缺点。激活函数中常见的一个问题是在反向传播时得到零梯度（或接近零的梯度）流。下列哪些激活函数会有这个问题？如果考虑一维情形，什么类型的输入会导致这种行为？
1. Sigmoid
2. ReLU
3. Leaky ReLU

$\\color{blue}{\\textit 你的回答：}$ *在此填写*""",
    ("assignment1/two_layer_net.ipynb", 14): """# “三明治”层
神经网络中经常会出现一些常用的层组合模式。例如，affine 层后面通常接一个 ReLU 非线性。为了方便使用这些常见模式，我们在 `cs231n/layer_utils.py` 文件中定义了几个便利层。

现在先查看 `affine_relu_forward` 和 `affine_relu_backward` 函数，然后运行下面的代码，对反向传播做数值梯度检查：""",
    ("assignment1/two_layer_net.ipynb", 16): """# 损失层：Softmax
现在在 `cs231n/layers.py` 的 `softmax_loss` 函数中实现 softmax 的损失和梯度。这应该与你在 `cs231n/classifiers/softmax.py` 中实现的内容类似。其他损失函数（例如 `svm_loss`）也可以用模块化方式实现，但本作业不要求。

运行下面的代码可以确认实现是否正确：""",
    ("assignment1/two_layer_net.ipynb", 18): """# 两层网络
打开 `cs231n/classifiers/fc_net.py` 文件，完成 `TwoLayerNet` 类的实现。通读代码，确保你理解它的 API。你可以运行下面的单元来测试实现。""",
    ("assignment1/two_layer_net.ipynb", 20): """# Solver
打开 `cs231n/solver.py` 文件并阅读代码，熟悉它的 API。之后，使用一个 `Solver` 实例训练 `TwoLayerNet`，使其在验证集上达到约 `36%` 的准确率。""",
    ("assignment1/two_layer_net.ipynb", 22): """# 调试训练过程
使用上面提供的默认参数时，你应该在验证集上得到约 0.36 的准确率。这并不算好。

理解问题的一种策略是在优化过程中绘制损失函数、训练集准确率和验证集准确率。

另一种策略是可视化网络第一层学到的权重。大多数在视觉数据上训练的神经网络，其第一层权重在可视化时通常会呈现某些可见结构。""",
    ("assignment1/two_layer_net.ipynb", 25): """# 调优超参数

**哪里不对？** 从上面的可视化可以看到，损失大致线性下降，这似乎表明学习率可能太低。此外，训练准确率和验证准确率之间没有差距，说明我们使用的模型容量较低，应该增大模型规模。另一方面，如果模型非常大，我们预期会看到更多过拟合，表现为训练准确率和验证准确率之间有很大的差距。

**调参。** 调整超参数并培养对它们如何影响最终性能的直觉，是使用神经网络的重要部分，所以我们希望你多加练习。下面你应该尝试不同的超参数取值，包括隐藏层大小、学习率、训练 epoch 数量和正则化强度。你也可以考虑调学习率衰减，不过使用默认值也应该能得到不错的性能。

**近似结果。** 你的目标是在验证集上达到高于 48% 的分类准确率。我们的最佳网络在验证集上超过 52%。

**实验：** 本练习的目标是用一个全连接神经网络在 CIFAR-10 上尽可能取得好结果（52% 可以作为参考）。你可以自由实现自己的技术，例如用 PCA 降维、添加 dropout、给 solver 增加功能等。""",
    ("assignment1/two_layer_net.ipynb", 27): """# 测试你的模型
在验证集和测试集上运行你的最佳模型。你应该在验证集和测试集上都达到 48% 以上的准确率。""",
    ("assignment1/two_layer_net.ipynb", 31): """## 内联问题 2：

现在你已经训练了一个神经网络分类器，可能会发现测试准确率远低于训练准确率。我们可以通过哪些方式缩小这个差距？请选择所有适用项。

1. 在更大的数据集上训练。
2. 添加更多隐藏单元。
3. 增大正则化强度。
4. 以上都不是。

$\\color{blue}{\\textit 你的回答：}$

$\\color{blue}{\\textit 你的解释：}$""",
}


TRANSLATIONS.update({
    ("assignment2/BatchNormalization.ipynb", 0): AI_USE,
    ("assignment2/BatchNormalization.ipynb", 1): STUDENT_DECLARATION,
    ("assignment2/BatchNormalization.ipynb", 3): """# Batch Normalization
让深度网络更容易训练的一种方法，是使用更复杂的优化过程，例如 SGD+momentum、RMSProp 或 Adam。另一种方法是改变网络结构，让它本身更易于训练。Batch normalization 就是这类思路中的一种方法，由文献 [1] 于 2015 年提出。

要理解 batch normalization 的目标，首先要认识到：机器学习方法通常在输入数据具有零均值、单位方差且特征之间相关性较低时表现更好。训练神经网络时，我们可以在把数据送入网络之前先做预处理，显式降低输入特征之间的相关性。这样能保证网络第一层看到的数据分布更友好。然而，即使输入数据已经预处理，网络深层的激活值也很可能不再满足低相关、零均值和单位方差，因为它们是前面层的输出。更糟的是，在训练过程中，每一层的权重不断更新，各层特征分布也会随之漂移。

文献 [1] 的作者假设，深度神经网络内部特征分布的漂移会让训练更加困难。为解决这个问题，他们提出在网络中插入对 batch 做归一化的层。训练时，这类层使用一个 minibatch 的数据估计每个特征的均值和标准差，并用这些估计值对 minibatch 的特征做中心化和归一化。训练过程中还会维护这些均值和标准差的滑动平均；测试时使用这些滑动平均来中心化和归一化特征。

这种归一化策略有可能降低网络的表达能力，因为某些层在最优情况下可能确实需要非零均值或非单位方差的特征。因此，batch normalization 层还为每个特征维度加入可学习的平移参数和缩放参数。

[1] [Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift", ICML 2015.](https://arxiv.org/abs/1502.03167)""",
    ("assignment2/BatchNormalization.ipynb", 6): """# Batch Normalization：前向传播
在 `cs231n/layers.py` 文件中，在 `batchnorm_forward` 函数里实现 batch normalization 的前向传播。完成后运行下面的代码测试你的实现。

上面 [1] 中链接的论文可能会有帮助。""",
    ("assignment2/BatchNormalization.ipynb", 9): """# Batch Normalization：反向传播
现在在 `batchnorm_backward` 函数中实现 batch normalization 的反向传播。

推导反向传播时，你应该写出 batch normalization 的计算图，并对每个中间节点做反向传播。有些中间量可能会流向多个分支；在反向传播中要确保把这些分支上的梯度相加。上面 [1] 中链接的论文可能会有帮助。

完成后，运行下面的代码对反向传播做数值检查。

_提示：[这个资源](https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm) 解释了如何从论文推导梯度。_""",
    ("assignment2/BatchNormalization.ipynb", 11): """# Batch Normalization：另一种反向传播实现
课上我们讨论过 sigmoid 反向传播的两种实现方式。一种策略是写出由简单操作组成的计算图，并对所有中间值做反向传播。另一种策略是在纸上直接推导导数。例如，你可以通过在纸上化简梯度，得到 sigmoid 函数反向传播的一个非常简洁公式。

有些令人意外的是，batch normalization 的反向传播也可以做类似化简。

在前向传播中，给定一组输入 $X=\\begin{bmatrix}x_1\\\\x_2\\\\...\\\\x_N\\end{bmatrix}$，

我们首先计算均值 $\\mu$ 和方差 $v$。有了 $\\mu$ 和 $v$ 后，可以计算标准差 $\\sigma$ 和归一化数据 $Y$。下面的公式和计算图说明了这个计算过程，其中 $y_i$ 是向量 $Y$ 的第 i 个元素。

\\begin{align}
& \\mu=\\frac{1}{N}\\sum_{k=1}^N x_k  &  v=\\frac{1}{N}\\sum_{k=1}^N (x_k-\\mu)^2 \\\\
& \\sigma=\\sqrt{v+\\epsilon}         &  y_i=\\frac{x_i-\\mu}{\\sigma}
\\end{align}""",
    ("assignment2/BatchNormalization.ipynb", 12): """<img src="https://raw.githubusercontent.com/cs231n/cs231n.github.io/master/assets/a2/batchnorm_graph.png">""",
    ("assignment2/BatchNormalization.ipynb", 13): """反向传播中的核心问题是：给定我们收到的上游梯度 $\\frac{\\partial L}{\\partial Y}$，计算 $\\frac{\\partial L}{\\partial X}$。根据微积分中的链式法则：
$\\frac{\\partial L}{\\partial X} = \\frac{\\partial L}{\\partial Y} \\cdot \\frac{\\partial Y}{\\partial X}$。

未知且困难的部分是 $\\frac{\\partial Y}{\\partial X}$。我们可以先逐步推导局部梯度：
$\\frac{\\partial v}{\\partial X}$、$\\frac{\\partial \\mu}{\\partial X}$、$\\frac{\\partial \\sigma}{\\partial v}$、$\\frac{\\partial Y}{\\partial \\sigma}$ 和 $\\frac{\\partial Y}{\\partial \\mu}$，然后用链式法则把这些梯度（它们会以向量形式出现）正确组合起来，计算 $\\frac{\\partial Y}{\\partial X}$。

如果直接对需要矩阵乘法的 $X$ 和 $Y$ 做梯度推理比较困难，可以先从单个元素 $x_i$ 和 $y_i$ 出发：这时你需要先用链式法则计算中间量 $\\frac{\\partial \\mu}{\\partial x_i}$、$\\frac{\\partial v}{\\partial x_i}$、$\\frac{\\partial \\sigma}{\\partial x_i}$，再把这些部分组合起来得到 $\\frac{\\partial y_i}{\\partial x_i}$。

为了便于实现，请尽量把每个中间梯度推导都化简到最简形式。

完成推导后，在 `batchnorm_backward_alt` 函数中实现简化版 batch normalization 反向传播，并运行下面的代码比较两种实现。两种实现应该得到几乎相同的结果，但替代实现应该稍快一些。

_提示：https://cs.stanford.edu/people/jcjohns/batchnorm.pdf。注意最终推导中应包含 γ。_

_请注意，这个 pdf 中公式 (8) 应写为：_ 
$$\\frac{1}{\\sigma} \\frac{\\partial}{\\partial x_i} (x_j - \\mu)
- \\frac{1}{\\sigma^2} \\frac{\\partial \\sigma}{\\partial x_i} (x_j - \\mu)$$""",
    ("assignment2/BatchNormalization.ipynb", 15): """# 带 Batch Normalization 的全连接网络
现在你已经有了可用的 batch normalization 实现，回到 `cs231n/classifiers/fc_net.py` 文件中的 `FullyConnectedNet`。回忆一下，你在 Assignment 1 中已经实现了网络初始化、前向传播和反向传播。把那份实现复制到这里，并修改它以加入 batch normalization。

具体来说，当构造函数中的 `normalization` 标志设置为 `"batchnorm"` 时，你应该在每个 ReLU 非线性之前插入一个 batch normalization 层。网络最后一层的输出不应该归一化。完成后，运行下面的代码对实现做梯度检查。

**提示：** 你可能会发现，定义一个类似 `cs231n/layer_utils.py` 中那些辅助层的额外 helper layer 会很方便。""",
    ("assignment2/BatchNormalization.ipynb", 17): """# 深层网络中的 Batch Normalization
运行下面的代码，在 1000 个训练样本的子集上分别训练有 batch normalization 和没有 batch normalization 的六层网络。""",
    ("assignment2/BatchNormalization.ipynb", 19): """运行下面的代码可视化上面训练的两个网络的结果。你应该会发现，使用 batch normalization 可以帮助网络收敛得更快。""",
    ("assignment2/BatchNormalization.ipynb", 21): """# Batch Normalization 与初始化
现在我们运行一个小实验，研究 batch normalization 和权重初始化之间的相互作用。

第一个单元会使用不同的权重初始化尺度，分别训练有 batch normalization 和没有 batch normalization 的八层网络。第二个单元会绘制训练准确率、验证集准确率和训练损失随权重初始化尺度变化的曲线。""",
    ("assignment2/BatchNormalization.ipynb", 24): """## 内联问题 1：
描述这个实验的结果。权重初始化尺度对有 batch normalization 和没有 batch normalization 的模型分别有什么不同影响？为什么？

## 回答：
[在此填写]""",
    ("assignment2/BatchNormalization.ipynb", 25): """# Batch Normalization 与 Batch Size
现在我们运行一个小实验，研究 batch normalization 和 batch size 之间的相互作用。

第一个单元会使用不同 batch size，分别训练有 batch normalization 和没有 batch normalization 的六层网络。第二个单元会绘制训练准确率和验证集准确率随时间变化的曲线。""",
    ("assignment2/BatchNormalization.ipynb", 28): """## 内联问题 2：
描述这个实验的结果。这说明 batch normalization 和 batch size 之间有什么关系？为什么会观察到这种关系？

## 回答：
[在此填写]""",
    ("assignment2/BatchNormalization.ipynb", 29): """# Layer Normalization
Batch normalization 已被证明能有效降低网络训练难度，但它依赖 batch size，因此在一些由于硬件限制而必须使用较小输入 batch size 的复杂网络中不太适用。

为缓解这个问题，人们提出了几种 batch normalization 的替代方法；其中一种是 Layer Normalization [2]。Layer Normalization 不在 batch 维度上归一化，而是在特征维度上归一化。换句话说，使用 Layer Normalization 时，每个单独数据点对应的特征向量会根据该特征向量内部所有项的统计量进行归一化。

[2] [Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." stat 1050 (2016): 21.](https://arxiv.org/pdf/1607.06450.pdf)""",
    ("assignment2/BatchNormalization.ipynb", 30): """## 内联问题 3：
下列哪些数据预处理步骤类似于 batch normalization，哪些类似于 layer normalization？

1. 对数据集中的每张图像进行缩放，使图像内每一行像素的 RGB 通道和为 1。
2. 对数据集中的每张图像进行缩放，使图像内所有像素的 RGB 通道总和为 1。
3. 从数据集中的每张图像中减去数据集的均值图像。
4. 根据给定阈值，把所有 RGB 值设置为 0 或 1。

## 回答：
[在此填写]""",
    ("assignment2/BatchNormalization.ipynb", 31): """# Layer Normalization：实现

现在你将实现 layer normalization。这一步应该相对直接，因为从概念上看，它的实现几乎与 batch normalization 相同。一个重要区别是：对于 layer normalization，我们不维护移动均值/方差；测试阶段和训练阶段相同，均值和方差都直接按每个数据点计算。

你需要完成：

* 在 `cs231n/layers.py` 中，在 `layernorm_forward` 函数里实现 layer normalization 的前向传播。

运行下面的单元检查结果。
* 在 `cs231n/layers.py` 中，在 `layernorm_backward` 函数里实现 layer normalization 的反向传播。

运行第二个单元检查结果。
* 修改 `cs231n/classifiers/fc_net.py`，为 `FullyConnectedNet` 添加 layer normalization。当构造函数中的 `normalization` 标志设置为 `"layernorm"` 时，你应该在每个 ReLU 非线性之前插入一个 layer normalization 层。

运行第三个单元，在 layer normalization 上执行 batch size 实验。""",
    ("assignment2/BatchNormalization.ipynb", 34): """# Layer Normalization 与 Batch Size

现在我们用 layer normalization 代替 batch normalization，重新运行前面的 batch size 实验。相比前一个实验，你应该看到 batch size 对训练历史的影响明显更小。""",
    ("assignment2/BatchNormalization.ipynb", 36): """## 内联问题 4：
Layer normalization 在什么情况下可能效果不好？为什么？

1. 在非常深的网络中使用。
2. 特征维度非常小。
3. 正则化项很大。

## 回答：
[在此填写]""",

    ("assignment2/ConvolutionalNetworks.ipynb", 0): AI_USE,
    ("assignment2/ConvolutionalNetworks.ipynb", 2): """# 卷积网络

到目前为止，我们一直使用深层全连接网络，用它们探索不同优化策略和网络架构。全连接网络由于计算效率较高，是很好的实验平台；但在实际应用中，所有最先进的结果基本都使用卷积网络。

首先你将实现卷积网络中会用到的几类层。然后你将用这些层在 CIFAR-10 数据集上训练一个卷积网络。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 5): """# 卷积：朴素前向传播
卷积网络的核心是卷积操作。在 `cs231n/layers.py` 文件中，在 `conv_forward_naive` 函数里实现卷积层的前向传播。

现在不需要太担心效率；用你觉得最清晰的方式写代码即可。

运行下面的代码可以测试你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 7): """## 插曲：用卷积做图像处理

为了检查你的实现，并更好理解卷积层能执行哪些类型的操作，我们会构造一个包含两张图像的输入，并手动设置一些滤波器来执行常见的图像处理操作（灰度转换和边缘检测）。卷积前向传播会把这些操作应用到每张输入图像上。随后我们可以可视化结果，作为合理性检查。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 9): """# 卷积：朴素反向传播
在 `cs231n/layers.py` 文件的 `conv_backward_naive` 函数中实现卷积操作的反向传播。同样，这里不需要太担心计算效率。

完成后，运行下面的代码，用数值梯度检查你的反向传播。

_提示：https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf 可以帮助你获得总体理解。实际的朴素实现会使用嵌套 for 循环，并利用来自 `dw` 和 `dout` 的单独导数直接计算 `dx_padded`。_""",
    ("assignment2/ConvolutionalNetworks.ipynb", 11): """# Max-Pooling：朴素前向传播
在 `cs231n/layers.py` 文件的 `max_pool_forward_naive` 函数中实现 max-pooling 操作的前向传播。同样，不必太担心计算效率。

运行下面的代码检查你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 13): """# Max-Pooling：朴素反向传播
在 `cs231n/layers.py` 文件的 `max_pool_backward_naive` 函数中实现 max-pooling 操作的反向传播。不需要担心计算效率。

运行下面的代码，用数值梯度检查你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 15): """# 快速层

让卷积层和池化层运行得很快并不容易。为减少这部分痛苦，我们已经在 `cs231n/fast_layers.py` 中提供了卷积和池化层前向、反向传播的快速实现。

### 执行下面的单元，保存 notebook，然后重启 runtime
快速卷积实现依赖一个 Cython 扩展；要编译它，请运行下面的单元。接着保存 Colab notebook（`File > Save`）并**重启 runtime**（`Runtime > Restart runtime`）。之后你可以从上到下重新执行前面的单元，并跳过下面这个单元，因为编译步骤只需要运行一次。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 17): """快速版卷积层和池化层的 API 与你上面实现的朴素版本完全相同：前向传播接收数据、权重和参数，产生输出和 cache 对象；反向传播接收上游导数和 cache 对象，产生关于数据和权重的梯度。

**注意：** 快速版 pooling 只有在 pooling 区域互不重叠且正好铺满输入时才会达到最优性能。如果不满足这些条件，快速 pooling 实现不会比朴素实现快太多。

运行下面的代码可以比较这些层朴素版本和快速版本的性能：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 20): """# 卷积 “三明治” 层
在上一个作业中，我们介绍了 “sandwich” 层的概念，它们把多个操作组合成常见模式。在 `cs231n/layer_utils.py` 文件中，你会找到一些为卷积网络常用模式实现的 sandwich 层。运行下面的单元，对它们的用法做合理性检查。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 23): """# 三层卷积网络
现在你已经实现了所有必要的层，可以把它们组合成一个简单的卷积网络。

打开 `cs231n/classifiers/cnn.py` 文件，完成 `ThreeLayerConvNet` 类的实现。记住，你可以在实现中使用 fast/sandwich 层（已经为你导入）。运行下面的单元帮助你调试：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 24): """## 合理性检查：损失
构建新网络后，首先应该检查损失是否合理。使用 softmax 损失时，随机权重（且无正则化）对应的损失大约应为 `log(C)`，其中 `C` 是类别数。加入正则化后，损失应该略有上升。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 26): """## 梯度检查
当损失看起来合理后，使用数值梯度检查确认反向传播是否正确。做数值梯度检查时，应该使用少量人工数据，并在每层使用较少神经元。注意：正确实现的相对误差仍可能达到 `1e-2` 数量级。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 28): """## 在小数据上过拟合
一个有用的技巧是只用少量训练样本训练模型。你应该能够让模型过拟合小数据集，这会带来很高的训练准确率和相对较低的验证准确率。""",
    ("assignment2/ConvolutionalNetworks.ipynb", 32): """绘制损失、训练准确率和验证准确率后，应该能清楚看到过拟合现象：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 34): """## 训练网络
训练三层卷积网络一个 epoch 后，你应该在训练集上达到高于 40% 的准确率：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 38): """## 可视化滤波器
运行下面的代码，可以可视化训练后网络第一层的卷积滤波器：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 40): """# Spatial Batch Normalization
我们已经看到 batch normalization 对训练深层全连接网络非常有用。正如原论文中提出的那样（链接见 `BatchNormalization.ipynb`），batch normalization 也可以用于卷积网络，但需要稍作调整；这个变体称为 “spatial batch normalization”。

普通 batch normalization 接收形状为 `(N, D)` 的输入，并输出形状为 `(N, D)` 的结果，其中我们沿 minibatch 维度 `N` 做归一化。对于来自卷积层的数据，batch normalization 需要接收形状为 `(N, C, H, W)` 的输入，并输出同样形状的结果；其中 `N` 表示 minibatch 大小，`(H, W)` 表示特征图的空间尺寸。

如果特征图由卷积产生，那么我们预期每个特征通道的统计量（例如均值和方差）在不同图像之间、同一图像的不同位置之间都相对一致。毕竟，每个特征通道都由同一个卷积滤波器产生。因此，spatial batch normalization 会对每个 `C` 特征通道，沿 minibatch 维度 `N` 以及空间维度 `H` 和 `W` 共同计算统计量。

[1] [Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift", ICML 2015.](https://arxiv.org/abs/1502.03167)""",
    ("assignment2/ConvolutionalNetworks.ipynb", 41): """# Spatial Batch Normalization：前向传播

在 `cs231n/layers.py` 文件中，在 `spatial_batchnorm_forward` 函数里实现 spatial batch normalization 的前向传播。运行下面的代码检查你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 44): """# Spatial Batch Normalization：反向传播
在 `cs231n/layers.py` 文件中，在 `spatial_batchnorm_backward` 函数里实现 spatial batch normalization 的反向传播。运行下面的代码，用数值梯度检查你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 46): """# Spatial Group Normalization
在前一个 notebook 中，我们提到 Layer Normalization 是一种可以缓解 Batch Normalization 受 batch size 限制影响的替代归一化技术。然而，正如文献 [2] 的作者观察到的那样，把 Layer Normalization 用在卷积层上时，它的表现不如 Batch Normalization：

> 在全连接层中，同一层的所有隐藏单元往往对最终预测作出相似贡献，因此对这一层的求和输入重新中心化和重新缩放通常效果不错。然而，在卷积神经网络中，“贡献相似” 的假设不再成立。大量位于图像边界附近感受野内的隐藏单元很少被激活，因此它们的统计量与同一层中其余隐藏单元非常不同。

文献 [3] 的作者提出了一种折中技术。与 Layer Normalization 对每个数据点的整个特征做归一化不同，他们建议把每个数据点的特征稳定地划分为 `G` 个组，并对每个数据点的每个组分别归一化。

<p align="center">
<img src="https://raw.githubusercontent.com/cs231n/cs231n.github.io/master/assets/a2/normalization.png">
</p>
<center>目前讨论过的归一化技术的视觉对比（图片由 [3] 修改而来）</center>

尽管每个组内仍然假设贡献相似，但作者认为问题没那么严重，因为视觉识别特征中天然存在分组结构。他们用来说明这一点的一个例子是：传统计算机视觉中很多高性能手工特征都会显式分组。例如方向梯度直方图（Histogram of Oriented Gradients, HOG）[4]：在每个局部空间块中计算直方图后，每个块的直方图会先归一化，再拼接成最终特征向量。

现在你将实现 Group Normalization。

[2] [Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." stat 1050 (2016): 21.](https://arxiv.org/pdf/1607.06450.pdf)

[3] [Wu, Yuxin, and Kaiming He. "Group Normalization." arXiv preprint arXiv:1803.08494 (2018).](https://arxiv.org/abs/1803.08494)

[4] [N. Dalal and B. Triggs. Histograms of oriented gradients for
human detection. In Computer Vision and Pattern Recognition
(CVPR), 2005.](https://ieeexplore.ieee.org/abstract/document/1467360/)""",
    ("assignment2/ConvolutionalNetworks.ipynb", 47): """# Spatial Group Normalization：前向传播

在 `cs231n/layers.py` 文件中，在 `spatial_groupnorm_forward` 函数里实现 group normalization 的前向传播。运行下面的代码检查你的实现：""",
    ("assignment2/ConvolutionalNetworks.ipynb", 49): """# Spatial Group Normalization：反向传播
在 `cs231n/layers.py` 文件中，在 `spatial_groupnorm_backward` 函数里实现 spatial group normalization 的反向传播。运行下面的代码，用数值梯度检查你的实现：""",

    ("assignment2/Dropout.ipynb", 0): AI_USE,
    ("assignment2/Dropout.ipynb", 2): """# Dropout
Dropout [1] 是一种正则化神经网络的技术：它在前向传播时随机把部分输出激活置为零。在这个练习中，你将实现一个 dropout 层，并修改你的全连接网络，使其可以选择性使用 dropout。

[1] [Geoffrey E. Hinton et al, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv 2012](https://arxiv.org/abs/1207.0580)""",
    ("assignment2/Dropout.ipynb", 5): """# Dropout：前向传播
在 `cs231n/layers.py` 文件中实现 dropout 的前向传播。由于 dropout 在训练和测试时行为不同，请确保两种模式都实现。

完成后，运行下面的单元测试你的实现。""",
    ("assignment2/Dropout.ipynb", 7): """# Dropout：反向传播
在 `cs231n/layers.py` 文件中实现 dropout 的反向传播。完成后，运行下面的单元，用数值梯度检查你的实现。""",
    ("assignment2/Dropout.ipynb", 9): """## 内联问题 1：
如果在 dropout 层中不把通过 inverse dropout 的值除以 `p`，会发生什么？为什么会这样？

## 回答：
[在此填写]""",
    ("assignment2/Dropout.ipynb", 10): """# 带 Dropout 的全连接网络
在 `cs231n/classifiers/fc_net.py` 文件中，修改你的实现以使用 dropout。具体来说，如果网络构造函数接收到的 `dropout_keep_ratio` 参数不是 1，那么网络应该在每个 ReLU 非线性之后立即添加一个 dropout 层。完成后，运行下面的代码对实现做数值梯度检查。""",
    ("assignment2/Dropout.ipynb", 12): """# 正则化实验
作为实验，我们将在 500 个训练样本上训练一组两层网络：一个不使用 dropout，另一个使用保留概率 0.25。然后我们会可视化这两个网络随时间变化的训练准确率和验证准确率。""",
    ("assignment2/Dropout.ipynb", 15): """## 内联问题 2：
比较使用 dropout 和不使用 dropout 时的验证准确率与训练准确率。你的结果说明 dropout 作为正则化器有什么作用？

## 回答：
[在此填写]""",
})


TRANSLATIONS.update({
    ("assignment2/PyTorch.ipynb", 0): AI_USE,
    ("assignment2/PyTorch.ipynb", 2): """# PyTorch 简介

在本次作业中，你已经写了大量代码，实现了许多神经网络功能。Dropout、Batch Norm 和二维卷积都是计算机视觉深度学习中的重要组件。你也花了很多精力让代码高效、向量化。

不过在本作业最后一部分，我们会暂时离开你亲手搭建的代码库，转向两个流行深度学习框架之一；这里使用 PyTorch。""",
    ("assignment2/PyTorch.ipynb", 3): """## 为什么使用深度学习框架？

* 我们的代码现在可以在 GPU 上运行。这会让模型训练快得多。使用 PyTorch 这样的框架时，你可以为自己的神经网络架构利用 GPU 能力，而不必直接编写 CUDA 代码（这超出了本课程范围）。
* 在本课程中，我们希望你能为最终项目做好准备，使用这些框架比手写每个所需功能更高效。
* 我们希望你站在前人成果之上。PyTorch 是很优秀的框架，会让你的工作轻松很多；现在你已经理解了这些组件内部的大致机制，可以放心使用它们。
* 最后，我们希望你接触到学术界或工业界中常见的深度学习代码形式。

## 什么是 PyTorch？

PyTorch 是一个在 Tensor 对象上执行动态计算图的系统；Tensor 的行为类似 numpy ndarray。它带有强大的自动微分引擎，因此不需要手写反向传播。

## 如何学习 PyTorch？

我们以前的一位课程讲师 Justin Johnson 为 PyTorch 制作了优秀的 [tutorial](https://github.com/jcjohnson/pytorch-examples)。

你也可以在这里找到详细的 [API doc](http://pytorch.org/docs/stable/index.html)。如果你有 API 文档没有覆盖的问题，[PyTorch forum](https://discuss.pytorch.org/) 通常比 StackOverflow 更适合提问。""",
    ("assignment2/PyTorch.ipynb", 4): """# 目录

本作业包含 5 个部分。你将从**三个不同抽象层次**学习 PyTorch，这有助于你更深入理解它，并为最终项目做准备。

1. Part I，准备：使用 CIFAR-10 数据集。
2. Part II，Barebones PyTorch：**抽象层次 1**，直接使用最底层的 PyTorch Tensor。
3. Part III，PyTorch Module API：**抽象层次 2**，使用 `nn.Module` 定义任意神经网络架构。
4. Part IV，PyTorch Sequential API：**抽象层次 3**，使用 `nn.Sequential` 非常方便地定义线性前馈网络。
5. Part V，CIFAR-10 开放挑战：实现你自己的网络，在 CIFAR-10 上尽可能取得高准确率。你可以尝试任意层、优化器、超参数或其他高级功能。

下面是一个对比表：

| API           | 灵活性 | 便利性 |
|---------------|--------|--------|
| Barebone      | 高     | 低     |
| `nn.Module`   | 高     | 中     |
| `nn.Sequential` | 低   | 高     |""",
    ("assignment2/PyTorch.ipynb", 5): """# GPU

在 Colab 中，你可以点击 `Runtime -> Change runtime type`，并在 `Hardware Accelerator` 下选择 `GPU`，手动切换到 GPU 设备。请在运行下面导入包的单元之前完成这一步，因为切换 runtime 会重启内核。""",
    ("assignment2/PyTorch.ipynb", 7): """# Part I：准备

现在加载 CIFAR-10 数据集。第一次执行可能需要几分钟，但文件之后应该会被缓存。

在本作业前面的部分中，我们需要自己写代码下载 CIFAR-10、预处理数据，并按 minibatch 迭代；PyTorch 提供了方便的工具，可以帮我们自动完成这些流程。""",
    ("assignment2/PyTorch.ipynb", 8): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 10): """# Part II：Barebones PyTorch

PyTorch 自带高层 API，可以方便地定义模型架构；这些会在本教程 Part III 中介绍。在本节中，我们先从 barebone PyTorch 元素开始，以便更好理解 autograd 引擎。完成这个练习后，你会更理解高层模型 API 的价值。

我们将从一个简单的全连接 ReLU 网络开始，它包含两个隐藏层、没有 bias，用于 CIFAR 分类。这个实现使用 PyTorch Tensor 运算计算前向传播，并使用 PyTorch autograd 计算梯度。你需要理解每一行代码，因为后面你会写一个更难的版本。

当我们创建一个 `requires_grad=True` 的 PyTorch Tensor 时，涉及这个 Tensor 的运算不仅会计算数值，还会在后台构建计算图，使我们能够方便地对计算图做反向传播，计算某些 Tensor 关于下游损失的梯度。具体来说，如果 `x` 是一个 Tensor 且 `x.requires_grad == True`，那么反向传播后 `x.grad` 会是另一个 Tensor，保存最终标量损失对 `x` 的梯度。""",
    ("assignment2/PyTorch.ipynb", 11): """### PyTorch Tensor：Flatten 函数
从概念上看，PyTorch Tensor 类似 numpy 数组：它是一个 n 维数字网格，并且像 numpy 一样提供许多高效操作 Tensor 的函数。作为简单例子，我们在下面提供一个 `flatten` 函数，它会把图像数据 reshape 成全连接神经网络可用的形式。

回忆一下，图像数据通常存储为形状 `N x C x H x W` 的 Tensor，其中：

* `N` 是数据点数量。
* `C` 是通道数。
* `H` 是中间特征图的像素高度。
* `W` 是中间特征图的像素宽度。

当我们做二维卷积等需要理解中间特征空间位置关系的操作时，这是表示数据的正确方式。但当我们使用全连接 affine 层处理图像时，希望每个数据点由一个单独向量表示；此时再区分数据的通道、行和列就不再有用。因此，我们使用 “flatten” 操作，把每个样本的 `C x H x W` 值折叠成一个长向量。下面的 flatten 函数先读取给定 batch 数据中的 `N`、`C`、`H` 和 `W`，然后返回这个数据的一个 “view”。“View” 类似 numpy 的 “reshape” 方法：它把 `x` 的维度改成 `N x ??`，其中 `??` 可以是任意值（这里是 `C x H x W`，但我们不需要显式指定）。""",
    ("assignment2/PyTorch.ipynb", 12): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 14): """### Barebones PyTorch：两层网络

这里定义 `two_layer_fc` 函数，它对一批图像数据执行两层全连接 ReLU 网络的前向传播。定义前向传播后，我们通过把全零输入传入网络来检查它不会崩溃，并且输出形状正确。

这里不需要你写代码，但阅读并理解实现很重要。""",
    ("assignment2/PyTorch.ipynb", 15): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 17): """### Barebones PyTorch：三层 ConvNet

这里你需要完成 `three_layer_convnet` 函数，它会执行三层卷积网络的前向传播。和上面类似，我们可以通过把全零输入传入网络，立即测试实现。网络架构如下：

1. 一个带 bias 的卷积层，包含 `channel_1` 个滤波器，每个滤波器形状为 `KW1 x KH1`，zero-padding 为 2。
2. ReLU 非线性。
3. 一个带 bias 的卷积层，包含 `channel_2` 个滤波器，每个滤波器形状为 `KW2 x KH2`，zero-padding 为 1。
4. ReLU 非线性。
5. 带 bias 的全连接层，输出 C 个类别的分数。

注意，全连接层之后这里**没有 softmax activation**：这是因为 PyTorch 的 cross entropy loss 会替你执行 softmax activation，并且把这一步合并进去可以提高计算效率。

**提示**：关于卷积请参考 http://pytorch.org/docs/stable/nn.html#torch.nn.functional.conv2d；注意卷积滤波器的形状。""",
    ("assignment2/PyTorch.ipynb", 18): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 20): """定义好上面 ConvNet 的前向传播后，运行下面的单元测试你的实现。

运行这个函数时，`scores` 的形状应为 `(64, 10)`。""",
    ("assignment2/PyTorch.ipynb", 21): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 23): """### Barebones PyTorch：初始化
我们来写几个工具方法，用于初始化模型的权重矩阵。

- `random_weight(shape)` 使用 Kaiming normalization 方法初始化权重 Tensor。
- `zero_weight(shape)` 初始化全零 Tensor，适合用于实例化 bias 参数。

`random_weight` 函数使用 Kaiming normal 初始化方法，见：

He et al, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, ICCV 2015, https://arxiv.org/abs/1502.01852""",
    ("assignment2/PyTorch.ipynb", 24): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 26): """### Barebones PyTorch：检查准确率
训练模型时，我们将使用下面的函数检查模型在训练集或验证集上的准确率。

检查准确率时不需要计算任何梯度，因此计算 scores 时也不需要 PyTorch 为我们构建计算图。为了阻止计算图被构建，我们把计算放在 `torch.no_grad()` context manager 中。""",
    ("assignment2/PyTorch.ipynb", 27): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 29): """### Barebones PyTorch：训练循环
现在可以搭建一个基础训练循环来训练网络。我们将使用不带 momentum 的随机梯度下降训练模型，并用 `torch.functional.cross_entropy` 计算损失；你可以在[这里阅读相关文档](http://pytorch.org/docs/stable/nn.html#cross-entropy)。

训练循环接收神经网络函数、初始化后的参数列表（例如例子中的 `[w1, w2]`）和学习率。""",
    ("assignment2/PyTorch.ipynb", 30): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 32): """### Barebones PyTorch：训练两层网络
现在可以运行训练循环了。我们需要显式分配全连接权重 `w1` 和 `w2` 的 Tensor。

CIFAR 的每个 minibatch 有 64 个样本，因此 Tensor 形状为 `[64, 3, 32, 32]`。

flatten 后，`x` 的形状应为 `[64, 3 * 32 * 32]`。这会是 `w1` 第一维的大小。`w1` 的第二维是隐藏层大小，同时也是 `w2` 第一维的大小。

最后，网络输出是一个 10 维向量，表示 10 个类别上的概率分布。

你不需要调任何超参数，但训练一个 epoch 后应该看到高于 40% 的准确率。""",
    ("assignment2/PyTorch.ipynb", 33): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 35): """### Barebones PyTorch：训练 ConvNet

下面你应该使用上面定义的函数，在 CIFAR 上训练一个三层卷积网络。网络架构如下：

1. 带 bias 的卷积层，包含 32 个 `5x5` 滤波器，zero-padding 为 2。
2. ReLU。
3. 带 bias 的卷积层，包含 16 个 `3x3` 滤波器，zero-padding 为 1。
4. ReLU。
5. 带 bias 的全连接层，计算 10 个类别的分数。

你应该使用上面定义的 `random_weight` 函数初始化权重矩阵，并使用 `zero_weight` 函数初始化 bias 向量。

你不需要调任何超参数；如果一切正常，训练一个 epoch 后应达到 42% 以上的准确率。""",
    ("assignment2/PyTorch.ipynb", 36): """<!-- 重要提示：为了保证你的解答能正确通过 autograder，请确保在解答中使用变量 `flat_feat_dim`。我们会用它确认你的维度是否正确，因此这一步非常重要。没有正确使用会导致得分为 0。 -->""",
    ("assignment2/PyTorch.ipynb", 38): """# Part III：PyTorch Module API

Barebone PyTorch 要求我们手动跟踪所有参数 Tensor。对于只有少数 Tensor 的小网络，这还可以接受；但对包含数十或数百个 Tensor 的大网络来说，这会非常不方便且容易出错。

PyTorch 提供了 `nn.Module` API，让你可以定义任意网络架构，同时自动跟踪所有可学习参数。在 Part II 中，我们自己实现了 SGD。PyTorch 还提供 `torch.optim` 包，实现了常见优化器，例如 RMSProp、Adagrad 和 Adam。它甚至支持 L-BFGS 这样的近似二阶方法。你可以参考 [doc](http://pytorch.org/docs/master/optim.html) 查看每个优化器的具体说明。

使用 Module API 时，请按以下步骤：

1. 继承 `nn.Module`。给你的网络类起一个直观名字，例如 `TwoLayerFC`。

2. 在构造函数 `__init__()` 中，把所需层定义为类属性。`nn.Linear` 和 `nn.Conv2d` 等层对象本身也是 `nn.Module` 的子类，并包含可学习参数，因此你不必自己实例化原始 Tensor。`nn.Module` 会为你跟踪这些内部参数。参考 [doc](http://pytorch.org/docs/master/nn.html) 了解更多内置层。**警告**：不要忘记先调用 `super().__init__()`。

3. 在 `forward()` 方法中定义网络的*连接关系*。你应该把 `__init__` 中定义的属性作为函数调用，接收 Tensor 输入并输出“变换后”的 Tensor。不要在 `forward()` 中创建任何带可学习参数的新层；它们都必须提前在 `__init__` 中声明。

定义好 Module 子类后，你可以实例化它，并像 Part II 中的神经网络前向函数一样调用它。

### Module API：两层网络
下面是一个两层全连接网络的具体例子：""",
    ("assignment2/PyTorch.ipynb", 39): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 41): """### Module API：三层 ConvNet
现在轮到你实现一个三层 ConvNet，并接一个全连接层。网络架构应与 Part II 相同：

1. 卷积层，包含 `channel_1` 个 `5x5` 滤波器，zero-padding 为 2。
2. ReLU。
3. 卷积层，包含 `channel_2` 个 `3x3` 滤波器，zero-padding 为 1。
4. ReLU。
5. 全连接层，输出 `num_classes` 个类别。

你应该使用 Kaiming normal 初始化方法初始化模型权重矩阵。

**提示**：http://pytorch.org/docs/stable/nn.html#conv2d

实现三层 ConvNet 后，`test_ThreeLayerConvNet` 函数会运行你的实现；它应该打印输出分数的形状 `(64, 10)`。""",
    ("assignment2/PyTorch.ipynb", 42): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 44): """### Module API：检查准确率
给定验证集或测试集，我们可以检查神经网络的分类准确率。

这个版本和 Part II 中的版本略有不同。你不再需要手动传入参数。""",
    ("assignment2/PyTorch.ipynb", 45): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 47): """### Module API：训练循环
这里也使用略有不同的训练循环。我们不再自己更新权重值，而是使用 `torch.optim` 包中的 Optimizer 对象；它抽象了优化算法这一概念，并提供了大多数常用神经网络优化算法的实现。""",
    ("assignment2/PyTorch.ipynb", 48): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 50): """### Module API：训练两层网络
现在可以运行训练循环了。与 Part II 不同，我们不再显式分配参数 Tensor。

只需把输入大小、隐藏层大小和类别数（即输出大小）传给 `TwoLayerFC` 的构造函数。

你还需要定义一个 optimizer，用来跟踪 `TwoLayerFC` 内部所有可学习参数。

你不需要调任何超参数，但训练一个 epoch 后应该看到高于 40% 的模型准确率。""",
    ("assignment2/PyTorch.ipynb", 51): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 53): """### Module API：训练三层 ConvNet
现在你应该使用 Module API 在 CIFAR 上训练一个三层 ConvNet。这应该和训练两层网络非常相似。你不需要调任何超参数，但训练一个 epoch 后应该达到 45% 以上的准确率。

请使用不带 momentum 的随机梯度下降训练模型。""",
    ("assignment2/PyTorch.ipynb", 54): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 56): """# Part IV：PyTorch Sequential API

Part III 介绍了 PyTorch Module API，它允许你定义任意可学习层及其连接关系。

对于简单模型，例如一串前馈层，你仍然需要经历 3 步：继承 `nn.Module`，在 `__init__` 中把层赋给类属性，并在 `forward()` 中逐个调用每一层。有没有更方便的方法？

幸运的是，PyTorch 提供了一个容器 Module，叫 `nn.Sequential`，它把上述步骤合并成一步。它不如 `nn.Module` 灵活，因为你不能指定比前馈堆叠更复杂的拓扑，但对许多场景来说已经足够。

### Sequential API：两层网络
我们来看如何用 `nn.Sequential` 重写两层全连接网络示例，并使用上面定义的训练循环来训练它。

同样，这里不需要调任何超参数，但训练一个 epoch 后应该达到 40% 以上的准确率。""",
    ("assignment2/PyTorch.ipynb", 57): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 59): """### Sequential API：三层 ConvNet
这里你应该使用 `nn.Sequential` 定义并训练一个三层 ConvNet，其架构与 Part III 中使用的相同：

1. 带 bias 的卷积层，包含 32 个 `5x5` 滤波器，zero-padding 为 2。
2. ReLU。
3. 带 bias 的卷积层，包含 16 个 `3x3` 滤波器，zero-padding 为 1。
4. ReLU。
5. 带 bias 的全连接层，计算 10 个类别的分数。

你可以使用 PyTorch 默认权重初始化。

请使用带 Nesterov momentum 0.9 的随机梯度下降优化模型。

同样，你不需要调任何超参数，但训练一个 epoch 后应该看到高于 55% 的准确率。""",
    ("assignment2/PyTorch.ipynb", 60): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 62): """# Part V：CIFAR-10 开放挑战

在这一节中，你可以在 CIFAR-10 上实验任何你喜欢的 ConvNet 架构。

现在你的任务是尝试不同架构、超参数、损失函数和优化器，训练一个在 10 个 epoch 内在 CIFAR-10 **验证集**上达到**至少 70%** 准确率的模型。你可以使用上面的 `check_accuracy` 和 `train` 函数，也可以使用 `nn.Module` 或 `nn.Sequential` API。

请在 notebook 末尾描述你做了什么。

下面是各组件的官方 API 文档。注意：课堂上称为 “spatial batch norm” 的东西，在 PyTorch 中叫 “BatchNorm2D”。

* torch.nn 包中的层：http://pytorch.org/docs/stable/nn.html
* 激活函数：http://pytorch.org/docs/stable/nn.html#non-linear-activations
* 损失函数：http://pytorch.org/docs/stable/nn.html#loss-functions
* 优化器：http://pytorch.org/docs/stable/optim.html

### 可以尝试的方向：
- **滤波器大小**：上面使用了 `5x5`；更小的滤波器会不会更高效？
- **滤波器数量**：上面使用了 32 个滤波器。更多或更少会不会更好？
- **Pooling vs Strided Convolution**：使用 max pooling，还是只使用 stride convolution？
- **Batch normalization**：尝试在卷积层后加入 spatial batch normalization，在 affine 层后加入普通 batch normalization。网络训练会更快吗？
- **网络架构**：上面的网络有两层可训练参数。使用更深网络能否做得更好？可以尝试的好架构包括：
    - `[conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]`
    - `[conv-relu-conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]`
    - `[batchnorm-relu-conv]xN -> [affine]xM -> [softmax or SVM]`
- **Global Average Pooling**：不要 flatten 后接多个 affine 层，而是持续卷积直到图像变小（例如 `7x7` 左右），然后执行 average pooling 得到 `1x1` 图像表示 `(1, 1, Filter#)`，再 reshape 成 `(Filter#)` 向量。这用在 [Google's Inception Network](https://arxiv.org/abs/1512.00567) 中（见其架构表 Table 1）。
- **正则化**：加入 L2 权重正则化，或使用 Dropout。

### 训练提示
对你尝试的每个网络架构，都应该调学习率和其他超参数。调参时请注意：

- 如果参数工作正常，几百次迭代内应该能看到提升。
- 记住超参数调优的 coarse-to-fine 方法：先用少量训练迭代测试较大范围的超参数，找到至少能工作的参数组合。
- 找到一些看起来有效的参数组合后，再围绕它们更精细地搜索。你可能需要训练更多 epoch。
- 应该使用验证集做超参数搜索，把测试集留到最后，用验证集选出的最佳参数评估你的架构。

### 进一步挑战
如果你想更进一步，还有很多功能可以实现来尝试提升性能。你**不需要**实现这些内容，但如果有时间也可以尝试。

- 替代优化器：Adam、Adagrad、RMSprop 等。
- 替代激活函数，例如 leaky ReLU、parametric ReLU、ELU 或 MaxOut。
- 模型集成。
- 数据增强。
- 新架构：
  - [ResNets](https://arxiv.org/abs/1512.03385)：把上一层输入加到输出上。
  - [DenseNets](https://arxiv.org/abs/1608.06993)：把前面层的输入拼接到一起。
  - [This blog has an in-depth overview](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32)

### 祝你训练顺利。""",
    ("assignment2/PyTorch.ipynb", 63): "<!-- -->",
    ("assignment2/PyTorch.ipynb", 65): """## 描述你做了什么

在下面的单元中，写下你做了哪些尝试、实现了哪些额外功能，以及训练和评估网络过程中画了哪些图。""",
    ("assignment2/PyTorch.ipynb", 66): """**回答：**""",
    ("assignment2/PyTorch.ipynb", 67): """## 测试集：只运行一次

现在我们已经得到了满意的结果，可以在测试集上测试最终模型（你应该把它存储在 `best_model` 中）。思考一下它与验证集准确率相比如何。""",
    ("assignment2/PyTorch.ipynb", 68): "<!-- -->",

    ("assignment2/RNN_Captioning_pytorch.ipynb", 0): AI_USE,
    ("assignment2/RNN_Captioning_pytorch.ipynb", 2): """# 使用 RNN 做图像描述
在这个练习中，你将实现普通循环神经网络（vanilla Recurrent Neural Networks），并用它们训练一个能够为图像生成新描述的模型。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 4): """# COCO 数据集
在这个练习中，我们将使用 2014 版 [COCO dataset](https://cocodataset.org/)，这是图像描述任务的标准测试平台。该数据集包含 80,000 张训练图像和 40,000 张验证图像，每张图像都由 Amazon Mechanical Turk 上的工作人员标注了 5 条描述。

**图像特征。** 我们已经为你预处理数据并提取好特征。对所有图像，我们从在 ImageNet 上预训练的 VGG-16 网络的 fc7 层提取特征，并将这些特征存储在 `train2014_vgg16_fc7.h5` 和 `val2014_vgg16_fc7.h5` 文件中。为了减少处理时间和内存需求，我们使用 Principal Component Analysis（PCA）把特征维度从 4096 降到 512，这些特征存储在 `train2014_vgg16_fc7_pca.h5` 和 `val2014_vgg16_fc7_pca.h5` 文件中。原始图像接近 20GB，因此没有包含在下载中。由于所有图像都来自 Flickr，我们把训练图像和验证图像的 URL 分别存储在 `train2014_urls.txt` 和 `val2014_urls.txt` 中。这样你可以在可视化时按需下载图像。

**描述。** 直接处理字符串效率较低，因此我们会使用编码后的描述。每个词都会被分配一个整数 ID，这样就可以把一条描述表示为整数序列。整数 ID 和单词之间的映射位于 `coco2014_vocab.json` 文件中，你可以使用 `cs231n/coco_utils.py` 文件中的 `decode_captions` 函数，把整数 ID 的 NumPy 数组转换回字符串。

**特殊 token。** 我们向词表中加入了几个特殊 token，并已经为你处理好相关实现细节。我们在每条描述开头添加特殊 `<START>` token，在结尾添加 `<END>` token。罕见词会被替换为特殊 `<UNK>` token（表示 “unknown”）。此外，由于训练 minibatch 中的描述长度不同，我们会在短描述的 `<END>` token 后填充特殊 `<NULL>` token，使所有描述长度相同，并且不会为 `<NULL>` token 计算损失或梯度。

你可以使用 `cs231n/coco_utils.py` 文件中的 `load_coco_data` 函数加载所有 COCO 数据（描述、特征、URL 和词表）。运行下面的单元执行加载：""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 6): """## 检查数据
开始处理数据集之前，查看一些示例总是好习惯。

你可以使用 `cs231n/coco_utils.py` 文件中的 `sample_coco_minibatch` 函数，从 `load_coco_data` 返回的数据结构中采样 minibatch。运行下面的代码，采样一个小的训练 minibatch，并显示图像及其描述。多运行几次并查看结果，有助于你理解数据集。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 8): """# 循环神经网络
如课上所述，我们将使用 Recurrent Neural Network（RNN）语言模型做图像描述。`cs231n/rnn_layers_pytorch.py` 文件包含循环神经网络所需的不同层类型实现，`cs231n/classifiers/rnn_pytorch.py` 文件使用这些层实现图像描述模型。

我们首先会在 `cs231n/rnn_layers_pytorch.py` 中实现不同类型的 RNN 层。

_如果想更深入理解 RNN，可选阅读：https://www.deeplearningbook.org/contents/rnn.html#pf7_""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 9): """# Vanilla RNN：单步前向传播
打开 `cs231n/rnn_layers_pytorch.py` 文件。该文件实现了循环神经网络常用不同层的前向传播。注意，由于这里使用 PyTorch，反向传播会由 PyTorch 的 autograd 处理。

首先实现 `rnn_step_forward` 函数，它实现 vanilla RNN 单个时间步的前向传播。完成后运行下面的代码检查你的实现。你应该看到 `1e-8` 数量级或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 11): """# Vanilla RNN：单步反向传播
由于我们用 PyTorch 实现了 `rnn_step_forward`，因此**不需要**实现 `rnn_step_backward`。我们可以用数值梯度检查器验证 PyTorch autograd 的反向传播。

不过，如果你想挑战自己，也可以尝试自己实现 `rnn_step_backward`。本作业不要求这一点。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 13): """# Vanilla RNN：前向传播
现在你已经实现了 vanilla RNN 单个时间步的前向传播，接下来要用它实现能处理整个序列数据的 RNN。

在 `cs231n/rnn_layers_pytorch.py` 文件中实现 `rnn_forward` 函数。它应该使用你上面定义的 `rnn_step_forward` 函数。完成后运行下面的代码检查你的实现。你应该看到 `1e-7` 数量级或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 15): """# Vanilla RNN：反向传播
和前面一样，我们可以使用数值梯度检查器验证 PyTorch autograd 的反向传播。如果你愿意，也可以自己尝试实现 `rnn_step_backward`，但本作业不要求。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 17): """# Word Embedding：前向传播
在深度学习系统中，我们通常用向量表示单词。词表中的每个单词都会关联一个向量，这些向量会和系统其余部分一起学习。

在 `cs231n/rnn_layers_pytorch.py` 文件中实现 `word_embedding_forward` 函数，把用整数表示的单词转换为向量。运行下面的代码检查你的实现。你应该看到 `1e-8` 数量级或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 19): """# Word Embedding：反向传播
和前面一样，我们可以使用数值梯度检查器验证 PyTorch autograd 的反向传播。如果你愿意，也可以尝试自己实现 `word_embedding_backward`，但本作业不要求。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 21): """# Temporal Affine 层
在每个时间步，我们使用 affine 函数把该时间步的 RNN hidden vector 转换为词表中每个单词的分数。由于这和你在 Assignment 1 中实现的 affine 层非常类似，我们已经在 `temporal_affine_forward` 中为你提供了这个函数。运行下面的代码对实现做数值梯度检查。你应该看到 `1e-9` 数量级或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 23): """# Temporal Softmax Loss
在 RNN 语言模型中，每个时间步都会为词表中的每个单词生成一个分数。我们知道每个时间步的真实单词，因此使用 softmax 损失函数计算该时间步的损失和梯度。我们把时间维度上的损失求和，并在 minibatch 上取平均。

但这里有一个细节：由于我们按 minibatch 操作，而不同描述长度可能不同，所以会在每条描述末尾追加 `<NULL>` token，使它们长度相同。我们不希望这些 `<NULL>` token 对损失或梯度有贡献，因此除了 scores 和真实标签外，损失函数还接收一个 `mask` 数组，用它说明 scores 中哪些元素应计入损失。

由于这和你在 Assignment 1 中实现的 softmax 损失函数非常类似，我们已经为你实现了这个损失函数；请查看 `cs231n/rnn_layers_pytorch.py` 文件中的 `temporal_softmax_loss` 函数。

运行下面的单元对损失做合理性检查，并对函数做数值梯度检查。你应该看到 `dx` 的误差在 `1e-7` 数量级或更小。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 25): """# 用 RNN 做图像描述
现在你已经实现了必要的层，可以把它们组合起来构建图像描述模型。打开 `cs231n/classifiers/rnn_pytorch.py` 文件，查看 `CaptioningRNN` 类。

在 `loss` 函数中实现模型的前向传播。目前只需要实现 `cell_type='rnn'` 的 vanilla RNN 情况；LSTM 情况会稍后实现。完成后，运行下面的代码，用一个小测试用例检查你的前向传播；你应该看到 `1e-10` 数量级或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 27): """运行下面的单元，对 `CaptioningRNN` 类做数值梯度检查；你应该看到 `1e-6` 数量级左右或更小的误差。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 29): """# 在小数据上过拟合 RNN 图像描述模型
类似于上一个作业中训练图像分类模型时使用的 `Solver` 类，本作业中我们使用 `CaptioningSolverPytorch` 类训练图像描述模型。打开 `cs231n/captioning_solver_pytorch.py` 文件，阅读 `CaptioningSolverPytorch` 类；它应该看起来很熟悉。

熟悉 API 后，运行下面的代码，确认你的模型能在 100 个训练样本的小样本上过拟合。最终损失应小于 0.1。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 31): """打印最终训练损失。你应该看到最终损失小于 0.1。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 33): """# 测试时的 RNN 采样
和分类模型不同，图像描述模型在训练时和测试时的行为差别很大。训练时，我们能访问真实描述，因此每个时间步都把真实单词作为输入喂给 RNN。测试时，我们在每个时间步从词表分布中采样，并把采样得到的单词作为下一时间步的输入喂给 RNN。

在 `cs231n/classifiers/rnn_pytorch.py` 文件中实现 `sample` 方法，用于测试时采样。完成后，运行下面的代码，在训练数据和验证数据上用你的过拟合模型采样。训练数据上的样本应该非常好；验证数据上的样本则很可能不太合理。""",
    ("assignment2/RNN_Captioning_pytorch.ipynb", 35): """# 内联问题 1

在当前图像描述设置中，我们的 RNN 语言模型在每个时间步输出一个单词。不过，也可以把问题改写为让网络在_字符_级别上工作（例如 'a'、'b' 等），而不是在单词级别上工作：每个时间步接收前一个字符作为输入，并尝试预测序列中的下一个字符。例如，网络可能生成如下描述：

'A', ' ', 'c', 'a', 't', ' ', 'o', 'n', ' ', 'a', ' ', 'b', 'e', 'd'

你能描述一个使用字符级 RNN 的图像描述模型的优点吗？也请描述一个缺点。提示：有多个合理答案，但比较词级模型和字符级模型的参数空间可能会有帮助。

**你的回答：**""",
    ("assignment2/collect_submission.ipynb", 1): """# 收集提交文件：压缩代码并生成 PDF

完成所有其他 notebook 后再运行这个 notebook：`BatchNormalization.ipynb`、`Dropout.ipynb`、`ConvolutionalNetworks.ipynb`、`PyTorch.ipynb` 和 `RNN_Captioning_pytorch.ipynb`。

它会：

* 生成一个包含你的代码（`.py` 和 `.ipynb`）的压缩包，文件名为 `a2.zip`。
* 将所有 notebook 转换成一个 PDF 文件，文件名为 `assignment.pdf`。

如果这一步提交文件生成成功，你应该会看到如下显示信息：

`### Done! Please submit a2.zip and the pdfs to Gradescope. ###`

请确保把 zip 和 pdf 文件下载到本地电脑，然后提交到 Gradescope。恭喜你完成本次作业。""",
})


TRANSLATIONS.update({
    ("assignment3/CLIP_DINO.ipynb", 0): AI_USE,
    ("assignment3/CLIP_DINO.ipynb", 4): """# 最先进的预训练图像模型

在前一个练习中，你学习了 [SimCLR](https://arxiv.org/abs/2002.05709)，以及如何使用对比式自监督学习来学习有意义的图像表示。在这个 notebook 中，我们会探索另外两个较新的模型；它们同样旨在学习高质量视觉表示，并且在多种下游任务上展示了强大且稳健的性能。

首先，我们会考察 [CLIP](https://github.com/openai/CLIP) 模型。和 SimCLR 一样，CLIP 使用对比学习目标；但它不是对比同一图像的两个增强视图，而是对比两种不同模态：文本和图像。为了训练 CLIP，OpenAI 从互联网上收集了约 4 亿个图像-文本对，包括 Wikipedia 和图像 alt text 等来源。最终模型学到了丰富的高层图像特征，并在许多视觉 benchmark 上取得了令人印象深刻的 zero-shot 性能。

接下来，我们会探索 [DINO](https://github.com/facebookresearch/dino)。它是一种用于视觉任务的自监督学习方法，在 self-distillation 框架和 multi-crop augmentation 策略中应用对比学习。作者表明，DINO ViT 学到的特征细粒度且语义丰富，并且显式包含图像语义分割相关信息。""",
    ("assignment3/CLIP_DINO.ipynb", 5): """# CLIP

如上所述，CLIP 的训练目标同时包含文本和图像，并建立在对比学习原则之上。回顾 SimCLR notebook 中的一句话：
> 对比损失的目标是最大化最终向量 **$z_i = g(h_i)$** 和 **$z_j = g(h_j)$** 之间的一致性。

类似地，CLIP 也被训练来最大化两个向量之间的一致性。不过，由于这些向量来自不同模态，CLIP 使用两个独立编码器：基于 Transformer 的 Text Encoder，以及基于 Vision Transformer（ViT）的 Image Encoder。注意，一些更小、更高效的 CLIP 版本会使用 ResNet 作为 Image Encoder，而不是 ViT。

运行下面的单元，可视化 CLIP 的训练和推理流程。

在预训练阶段，每个 batch 包含多张图像及其对应 caption。每张图像都会被 Image Encoder 独立处理，通常是 Vision Transformer（ViT）或 Convolutional Neural Network（ConvNet）这样的视觉模型，输出图像 embedding $I_n$。同样，每条 caption 会被 Text Encoder 独立处理，生成对应的文本 embedding $T_n$。接下来，我们计算所有图像-文本组合之间的成对相似度，也就是每张图像都与每条 caption 比较，反之亦然。训练目标是最大化所得相似度矩阵对角线上的分数，也就是匹配图像-caption 对 $(I_n, T_n)$ 的分数。通过反向传播，模型会学习给真实匹配分配比错误匹配更高的相似度。

通过这种设置，CLIP 能有效地把图像和文本表示到共享潜在空间中。在这个空间里，语义概念以一种与模态无关的方式编码，从而支持视觉输入和文本输入之间有意义的跨模态比较。""",
    ("assignment3/CLIP_DINO.ipynb", 7): """**内联问题 1** -

为什么 CLIP 的学习依赖 batch size？如果 batch size 固定，我们可以使用什么策略来学习丰富的图像特征？

$\\color{blue}{\\textit 你的回答：}$""",
    ("assignment3/CLIP_DINO.ipynb", 8): """# 加载 COCO 数据集

我们会使用你训练 RNN 图像描述模型时用过的同一个 captioning 数据集；但这次不是生成 caption，而是看看能否把每张图像匹配到正确的 caption。""",
    ("assignment3/CLIP_DINO.ipynb", 15): """# 运行 CLIP 模型

首先，我们会使用预训练 CLIP 模型分别从文本和图像中提取特征。""",
    ("assignment3/CLIP_DINO.ipynb", 20): """打开 `cs231n/clip_dino.py`，实现 `get_similarity_no_loop`，用于计算文本特征和图像特征之间的相似度分数。运行下面的代码测试你的实现；你应该看到小于 `1e-5` 的相对误差。""",
    ("assignment3/CLIP_DINO.ipynb", 23): """# Zero Shot 分类器

你会在上面看到匹配的图像-caption 对具有较高相似度。我们可以利用这个性质设计一个不需要任何标注数据的图像分类器，也就是 zero-shot classifier。每个类别可以用合适的自然语言描述来表示，任意输入图像都会被分类到 CLIP embedding 空间中与该图像相似度最高的描述所对应的类别。""",
    ("assignment3/CLIP_DINO.ipynb", 24): """在 `cs231n/clip_dino.py` 中实现 `clip_zero_shot_classifier`，并在下面测试。你应该能看到如下预测：

['a person', 'an animal', 'an animal', 'food', 'a person', 'a landscape', 'other', 'other', 'other', 'a person']""",
    ("assignment3/CLIP_DINO.ipynb", 26): """运行下面的单元可视化预测结果。可以看到，CLIP 提供了一种直接的方法，可以在任意类别体系上进行相当合理的 zero-shot 分类。

CLIP 是第一个在不使用任何 ImageNet 图像或标签的情况下，在 ImageNet 分类上超过标准监督训练的模型。原始 CLIP 论文中还有许多类似的有趣实验和分析。""",
    ("assignment3/CLIP_DINO.ipynb", 28): """# 使用 CLIP 做图像检索

正如我们使用 CLIP 为每张图像检索匹配类别名一样，也可以使用它根据文本输入检索匹配图像，也就是 semantic image retrieval。在 `cs231n/clip_dino.py` 中实现 `CLIPImageRetriever`，并运行下面两个单元进行测试。每个查询期望的 top 2 输出已在注释中给出。""",
    ("assignment3/CLIP_DINO.ipynb", 31): """**内联问题 2** -

CLIP 使用对比损失在共享潜在空间中对齐图像和文本表示。你会如何把这个思路扩展到两个以上的模态？

$\\color{blue}{\\textit 你的回答：}$""",
    ("assignment3/CLIP_DINO.ipynb", 32): """# DINO

如前所述，使用普通对比学习方法训练的模型（例如 SimCLR 和 CLIP）需要非常大的 batch size。这使它们计算代价高，并限制了可访问性。后续工作如 [BYOL](https://arxiv.org/abs/2006.07733) 提出了一种替代方法：用 student-teacher 框架避免大量负样本需求。这种方法效果出人意料地好，后来被 [DINO](https://arxiv.org/abs/2104.14294) 采用。

和 SimCLR 类似，DINO 被训练来最大化同一图像不同视图产生的两个向量之间的一致性。不过不同于 SimCLR，DINO 使用两个独立编码器，并且它们的训练方式不同。student network 通过反向传播更新，使其输出匹配 teacher network 的输出。teacher network 不通过反向传播更新；它的权重使用 student 权重的指数滑动平均（EMA）更新。这意味着 teacher 模型变化更慢，可以为 student 提供稳定的学习目标。

运行下面的单元可视化 DINO 的训练流程。""",
    ("assignment3/CLIP_DINO.ipynb", 37): """# DINO Attention Maps

由于加载的 DINO checkpoint 基于 ViT 架构，我们可以可视化每个 attention head 关注的内容。下面的代码会生成热力图，显示最终层中各个 head 的 [CLS] token 关注原始图像中的哪些 patch。虽然这个模型用自监督目标训练，并没有显式要求识别图像中的 “结构”，但是仍然会出现一些模式。

你注意到了什么规律吗？""",
    ("assignment3/CLIP_DINO.ipynb", 40): """**内联问题 3**

上面打印出的 tensor shape 是如何得到的？请解释你的答案。

$\\color{blue}{\\textit 你的回答：}$""",
    ("assignment3/CLIP_DINO.ipynb", 41): """# DINO 特征

为了理解模型在每个 patch 中编码了什么，我们可以可视化每个 patch token 的内容。由于这些 embedding 维度很高，难以直接解释，我们会使用 PCA 找到特征空间中方差最大的方向。

在下一个单元中，我们可视化特征空间中的三个主方差方向。这会揭示 patch embedding 捕捉到的主导结构。""",
    ("assignment3/CLIP_DINO.ipynb", 43): """**内联问题 4** -

你在上面的可视化中看到了什么样的结构？如果某个区域持续呈现特定颜色，这意味着什么？如果两个区域颜色明显不同，又意味着什么？请记住，PCA 揭示的是所有 patch 的特征空间中方差最大的方向。一个 patch 的颜色反映了它不同的特征内容。

$\\color{blue}{\\textit 你的回答：}$""",
    ("assignment3/CLIP_DINO.ipynb", 44): """# 基于 DINO 特征的简单分割模型

在上一节中，我们看到 DINO 特征可以提供出人意料地好的分割线索。现在，让我们在 [DAVIS dataset](https://davischallenge.org) 上训练一个简单分割模型来验证这个想法。DAVIS 数据集（Densely Annotated VIdeo Segmentation）是为视频目标分割任务创建的。它为视频中的目标提供逐帧、像素级标注。在这个实验中，我们只使用一个视频中单帧的标注来训练模型，并观察它在同一视频剩余帧上的表现。

我们的模型会刻意保持最小化：我们会按 patch 提取 DINO 特征，并只使用那一帧有标注图像中的 patch，训练一个轻量级 per-patch 分类器。通常情况下，你会在完整数据集上训练，并在包含不同视频的独立验证集上评估。但在这里，我们会测试 DINO 特征的 one-shot 能力。""",
    ("assignment3/CLIP_DINO.ipynb", 47): """完成 `cs231n/clip_dino.py` 中 `DINOSegmentation` 类的实现，并运行下面两个单元进行测试。你应该在第一个测试帧上达到大于 0.45 的 mean IoU，在最后一个测试帧上达到大于 0.50 的 mean IoU。为了避免在训练 patch 特征上过拟合，可以考虑设计一个非常轻量的模型，例如线性层或两层 MLP，并使用合适的 weight decay。

你可以使用 GPU runtime 加速训练和评估。如果切换 runtime type，请确保重新运行整个 notebook。""",
    ("assignment3/CLIP_DINO.ipynb", 50): """现在可视化结果。运行下面两个单元，显示第一帧、中间帧和最后一帧的 ground truth 与预测分割 mask。注意，中间帧属于训练集，而其他帧是未见过的。""",
    ("assignment3/CLIP_DINO.ipynb", 53): """现在运行下面三个单元，对整个视频进行评估和可视化。你应该达到大于 0.55 的 mean IoU。保存的可视化视频在 Google Drive 中处理可能需要一些时间，但你可以把它下载到电脑上本地查看。""",
    ("assignment3/CLIP_DINO.ipynb", 57): """**内联问题 5** -

如果你在 CLIP ViT 的 patch 特征上训练分割模型，你预期它会比 DINO 表现更好还是更差？为什么应该是这样？

$\\color{blue}{\\textit 你的回答：}$""",

    ("assignment3/DDPM.ipynb", 0): AI_USE,
    ("assignment3/DDPM.ipynb", 2): """# Denoising Diffusion Probabilistic Models

到目前为止，我们探索的都是判别式模型，它们被训练来产生带标签的输出。这些任务包括直接的图像分类，也包括句子生成；后者仍然可以被看作分类问题，只是标签位于词表空间，并用循环机制捕捉多词标签。现在，我们将扩展工具箱，构建一个生成式模型，使其能够生成与给定训练图像集合相似的新图像。

生成式模型有很多类型，包括 Generative Adversarial Networks（GANs）、autoregressive models、normalizing flow models 和 Variational Autoencoders（VAEs），它们都能合成令人印象深刻的图像。不过在 2020 年，Ho 等人通过把 diffusion probabilistic models 与 denoising score matching 结合，提出了 Denoising Diffusion Probabilistic Models（DDPMs）。这产生了一种既容易训练、又足以生成复杂高质量图像的生成模型。下面会给出 DDPM 的高层概览。更多细节请参考课程 slides 和原始 DDPM 论文 [1]。

# Forward Process
设 $q(x_0)$ 为干净数据集图像的分布。我们把前向加噪过程定义为一条由小加噪步骤组成的 Markov chain：

$$q(x_t | x_{t-1}) \\sim N(x_t; \\sqrt{1-\\beta_t} x_{t-1} , \\beta_t I)$$

其中逐步方差 $(\\beta_1, ..., \\beta_T)$ 决定 noise schedule。由于 Gaussian distribution 的性质，我们可以把 $q(x_t | x_0)$ 写成闭式形式：

$$q(x_t | x_0) \\sim N(x_t; \\sqrt{\\bar{\\alpha}_t} x_0 , (1-\\bar{\\alpha}_t) I)$$

其中 $\\alpha_t = 1-\\beta_t$，且 $\\bar{\\alpha}_t = \\prod_{s=1}^{t}\\alpha_t$。如果 noise schedule $(\\beta_1, ..., \\beta_T)$ 设置合适，最终分布 $q(x_T)$ 会变得与纯 Gaussian noise $N(0, I)$ 难以区分。

回忆一下，从 Gaussian distribution $x \\sim N(\\mu, \\sigma^2)$ 采样等价于计算 $\\sigma * \\epsilon + \\mu$，其中 $\\epsilon \\sim N(0, 1)$。因此，只要给定 $x_{t-1}$ 或 $x_0$，从 $q(x_t | x_{t-1})$ 或 $q(x_t | x_0)$ 采样都很直接。正因如此，前向过程很简单，不需要学习。

# Reverse Process
反向过程通过多个步骤，从纯噪声 $x_T$ 重建干净图像 $x_0$。令 $p(x_{t-1} | x_t)$ 表示 $q(x_t | x_{t-1})$ 的反向步骤。
第一个关键洞见是：学习反转每一个单独的去噪步骤，比一次性反转整个前向过程更容易。换句话说，对每个 $t$ 学习 $p(x_{t-1} | x_t)$，比直接学习 $p(x_0 | x_T)$ 更容易。

不过，学习 $p(x_{t-1} | x_t)$ 仍然有挑战。尽管 $q(x_t | x_{t-1})$ 是 Gaussian，$p(x_{t-1} | x_t)$ 可能具有任意复杂形式，而且几乎肯定不是 Gaussian。对任意分布建模并采样，远比处理 Gaussian 这样的简单参数化分布困难。

第二个关键洞见是：如果前向过程中的逐步噪声 $\\beta_t$ 足够小，那么反向步骤 $p(x_{t-1} | x_t)$ 也会接近 Gaussian distribution。因此，我们只需要估计它的均值和方差。实践中，把 $p(x_{t-1} | x_t)$ 的方差设置为匹配 $\\beta_t$（与前向步骤相同）效果很好。于是，学习反向过程就简化为学习均值 $\\mu(x_t, t; \\theta)$，其中 $\\theta$ 表示神经网络参数。

# Denoising Objective
生成式模型通常通过最小化数据集样本的期望负对数似然 $\\mathbb{E}[-\\log{p_\\theta(x_0)}]$ 来优化。每个样本的 likelihood 可以写为：$p_\\theta(x_0) = p(x_T)\\prod_{t=1}^T p(x_{t-1} | x_t)$。由于这个目标在许多情况下不可处理，不同类别的生成式模型会优化负对数似然的 variational lower bound。

Ho 等人证明，这个目标等价于最小化如下去噪损失：

$$\\mathbb{E}_{t, x_0, \\epsilon}\\left[ \\| \\epsilon - \\epsilon_\\theta (\\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, t ) \\|^2 \\right]$$

其中 $t$ 在 1 到 T 之间均匀采样，$x_0$ 是干净样本，$\\epsilon$ 从标准 Gaussian $N(0, I)$ 采样，$\\epsilon_\\theta$ 是一个神经网络模型，训练目标是从输入 noisy sample $x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$ 中预测噪声 $\\epsilon$。换句话说，$\\epsilon_\\theta$ 学会对输入 noisy image 去噪。注意，这等价于预测干净样本，因为根据等式 $x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$，噪声可以由 noisy image 和 clean sample 恢复。

[1] Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbeel. [Link](https://arxiv.org/pdf/2006.11239)""",
    ("assignment3/DDPM.ipynb", 3): """# 本 Notebook 的内容
我们将实现并训练一个 DDPM 模型，用于根据文本 prompt 生成小尺寸 `32 x 32` emoji 图像。首先，我们会根据论文 [1] 的公式 (4) 实现前向加噪过程。然后构建一个 UNet 模型，它接收 $x_t$ 和 $t$ 作为输入（也可以附加文本 prompt 等条件），并输出与 $x_t$ 形状相同的 tensor。最后，我们会实现去噪目标，并训练 DDPM 模型。

我们使用预训练 CLIP [2] 模型的 text encoder，将输入文本编码为 512 维向量。为了加快训练，我们已经提前对训练集中的文本数据做了编码。

[2] Learning transferable visual models from natural language supervision. Radford et. al. [Link](https://github.com/openai/CLIP)""",
    ("assignment3/DDPM.ipynb", 8): """## q_sample

现在定义前向加噪过程。阅读 `cs231n/gaussian_diffusion.py` 中的 `GaussianDiffusion` 类。公式请参考原始 DDPM 论文 [1]。实现 `q_sample` 方法，并在下面测试。你应该看到相对误差为 0。""",
    ("assignment3/DDPM.ipynb", 11): """扩散模型可以被训练来预测干净图像，也可以预测噪声，因为二者可以相互推导（见上面 “Denoising Objective” 部分）。实现 `predict_start_from_noise` 和 `predict_noise_from_start` 方法，并在下面测试。你应该看到小于 `1e-5` 的相对误差。""",
    ("assignment3/DDPM.ipynb", 13): """## UNet 模型

现在已经定义了前向过程，接下来定义用于输入图像去噪的 UNet 模型。UNet 是一种用于 image-to-image 任务的神经网络架构，例如分割、风格迁移等。它包含一个 encoder（或 downsampling module），把输入图像转换成空间分辨率逐渐降低、特征维度逐渐增大的层级特征。decoder（或 upsampling module）随后逐步恢复空间分辨率，并镜像 encoder 结构。在每个 decoder 层中，来自对应 encoder 层的特征会被拼接进来，为细节提供直接路径。这种方式减轻了 bottleneck 层的负担，让它们专注于捕捉高层表示，而不是记忆细粒度细节。

这里使用 UNet，是因为我们的输入和输出都是尺寸对齐的图像，维度相同：`C x H x W`。UNet 中的每个 ResNet block 还会接收一个额外输入向量，称为 context，用于条件控制。我们会通过编码 diffusion timestep 和文本 prompt 来生成 context vector。

运行下面的单元，粗略查看 UNet 架构。每个红色框表示一个 ResNet block，其中包含 2 或 3 个卷积层，并保持特征图空间分辨率不变。为简洁起见，图中省略了输入到每个 ResNet block 的 context vector。每个框下方写出的形状表示该 block 之后的输出 tensor 形状。额外箭头表示 skip connections，它们让 U-Net 能在输出中保留细粒度细节。例如，形状为 `(d, h, w)` 的 `layer1_block1` 输出会与同样形状为 `(d, h, w)` 的 `layer4_block1` 输出拼接，然后传给 `layer4_block2`。因此，`layer4_block2` 会接收形状为 `(2*d, h, w)` 的输入。""",
    ("assignment3/DDPM.ipynb", 15): """在 `cs231n/unet.py` 中实现 `Unet.__init__` 方法，以定义 UNet 模型的上采样和下采样 block，然后在下面测试。如果实现正确，你不应该看到任何错误。调用 `Unet(dim=d, condition_dim=condition_dim, dim_mults=(2,4))` 应该能成功创建与上图架构对应的 UNet 模型。""",
    ("assignment3/DDPM.ipynb", 17): """补全 `cs231n/unet.py` 中的 `Unet.forward` 方法，并在下面测试。暂时不用担心 `Unet.cfg_forward` 方法。你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/DDPM.ipynb", 19): """# p_losses

现在模型实现已经完成，接下来编写 DDPM 的去噪训练步骤。如前所述，优化去噪损失等价于最小化数据集的期望负对数似然。完成 `cs231n/gaussian_diffusion.py` 中的 `GaussianDiffusion.p_losses` 方法，并在下面测试。你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/DDPM.ipynb", 21): """## p_sample

现在还剩最后一个组成部分。DDPM 通过迭代执行反向过程来生成样本。这个反向过程的每一次迭代都涉及从 $p(x_{t-1}|x_t)$ 采样。打开 `cs231n/gaussian_diffusion.py`，按照论文公式 (6) 实现 `p_sample` 方法。该公式描述了在给定 $x_t$ 和 $x_0$ 条件下，从前向过程 posterior 中采样；其中 $x_0$ 可以由去噪模型的输出推导得到。我们已经实现了 `sample` 方法，它会迭代调用 `p_sample`，根据输入文本生成图像。

在下面测试你的 `p_sample` 实现；你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/DDPM.ipynb", 23): """## 训练

现在 DDPM 训练所需的所有组件都已具备，可以在 Emoji 数据集上训练模型。这里不需要你写代码，但建议你查看 `cs231n/ddpm_trainer.py` 中的训练代码。

在 notebook 剩余部分，我们会使用 `cs231n/exp/pretrained` 文件夹中的预训练模型；它已经在这个数据集上训练了许多迭代。不过你也可以在 Colab GPU 上训练自己的模型（确保修改 `results_folder`）。注意，在 T4 GPU 上可能需要超过 12 小时才能开始看到比较合理的生成结果。""",
    ("assignment3/DDPM.ipynb", 27): """## 采样

运行下面的单元，可视化由文本 prompt 条件控制的 emoji 生成结果。你可以自由修改 prompt，探索不同生成结果。由于我们的 emoji 数据集很小，不足以训练一个完全可泛化的 text-to-image 模型。因此，对未见过 prompt 的生成结果可能较差，也可能不忠实于输入文本（对见过的样本也可能发生，但较少见）。

为了加快采样，可以使用 GPU runtime。如果切换 runtime type，请确保重新运行整个 notebook。""",
    ("assignment3/DDPM.ipynb", 29): """## Classifier Free Guidance

生成式模型通常根据 fidelity（生成样本的质量或真实感）和 diversity（样本空间的变化性或覆盖度）来评估。对于条件生成模型，fidelity 还包括生成样本是否忠实遵循输入条件。这两个指标经常互相拉扯，形成 trade-off。Ho 等人提出了一个简单技术，叫 classifier-free guidance [3]，可以显式控制这种 trade-off。

在 classifier-free guidance 中，训练条件扩散模型 $\\epsilon_\\theta(x_t, t, c)$ 时，会以某个概率（通常为 0.1 到 0.2）随机丢弃条件 $c$，也就是替换为 $c=\\phi$。采样的每一步去噪中，预测会更新为：
$$\\epsilon_\\theta(x_t, t, c) \\leftarrow (w+1) \\epsilon_\\theta(x_t, t, c) - w \\epsilon_\\theta(x_t, t, \\phi)$$
其中 $w$ 是正标量，即 guidance scale。换句话说，我们在每个去噪步骤中做两次预测，一次有条件、一次无条件，并把它们线性组合以偏向条件生成。$w$ 是一个超参数，会根据具体模型的评估指标调优。更高的 $w$ 会让生成结果更忠实于条件，但通常会降低多样性。

[3] Classifier-Free Diffusion Guidance. Jonathan Ho, Tim Salimans. [Link](https://arxiv.org/abs/2207.12598)""",
    ("assignment3/DDPM.ipynb", 30): """在 `cs231n/unet.py` 的 `Unet.cfg_forward` 方法中实现 classifier-free guidance，并在下面测试。你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/DDPM.ipynb", 32): """运行下面的单元，使用 classifier-free guidance 可视化 emoji 生成结果。你也可以自由修改 `"cfg_scale"` 参数值。如前所述，由于我们的模型泛化能力有限，即使 guidance scale 很高，也不一定能观察到忠实的生成结果。""",
})


TRANSLATIONS.update({
    ("assignment3/Self_Supervised_Learning.ipynb", 0): AI_USE,
    ("assignment3/Self_Supervised_Learning.ipynb", 2): """## 使用 GPU

进入 `Runtime > Change runtime type`，将 `Hardware accelerator` 设置为 `GPU`。这会重置 Colab。**请重新运行顶部单元，再次挂载你的 Drive。**""",
    ("assignment3/Self_Supervised_Learning.ipynb", 3): """# 自监督学习

## 什么是自监督学习？
现代机器学习需要大量标注数据。但很多时候，获取大量人工标注数据既困难又昂贵。有没有办法让机器在没有标注数据集的情况下，自动学习一个能生成良好视觉表示的模型？有，这就是自监督学习。

自监督学习（SSL）允许模型在不使用标签的情况下，仅利用给定数据集中的数据自动学习一个“好”的表示空间。具体来说，如果我们的数据集是一组图像，自监督学习就能让模型为图像学习并生成“好”的表示向量。

SSL 方法近年来越来越受欢迎，原因是学到的模型在其他数据集上也能继续表现良好，也就是模型没有训练过的新数据集上也能迁移。

## 什么是“好”的表示？
一个“好”的表示向量需要捕捉图像中与整个数据集相关的重要特征。这意味着，数据集中表示语义相似实体的图像应该有相似的表示向量，而不同图像应该有不同的表示向量。例如，两张苹果图像应有相似表示向量，而苹果图像和香蕉图像应有不同表示向量。

## 对比学习：SimCLR
最近，[SimCLR](https://arxiv.org/pdf/2002.05709.pdf) 提出了一种使用**对比学习**来学习良好视觉表示的新架构。对比学习旨在为相似图像学习相似表示，为不同图像学习不同表示。正如我们将在这个 notebook 中看到的，这个简单想法让我们无需任何标签也能训练出相当不错的模型。

具体来说，对数据集中的每张图像，SimCLR 会生成该图像的两个不同增强视图，称为**正样本对**。然后，模型被鼓励为这一对图像生成相似的表示向量。下面是该架构的示意图（来自论文 Figure 2）。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 5): """给定一张图像 **x**，SimCLR 使用两种不同的数据增强方案 **t** 和 **t'** 生成正样本对图像 **$\\tilde{x}_i$** 和 **$\\tilde{x}_j$**。$f$ 是一个基础 encoder network，它从增强后的数据样本中提取表示向量，分别得到 **$h_i$** 和 **$h_j$**。最后，一个小型神经网络 projection head $g$ 会把表示向量映射到应用对比损失的空间。对比损失的目标是最大化最终向量 **$z_i = g(h_i)$** 和 **$z_j = g(h_j)$** 之间的一致性。稍后我们会更详细讨论对比损失，并由你来实现它。

训练完成后，我们丢弃 projection head $g$，只使用 $f$ 和表示 $h$ 执行下游任务，例如分类。你会有机会在训练好的 SimCLR 模型之上微调一个层来完成分类任务，并把它的性能与 baseline 模型（没有自监督学习）进行比较。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 6): """## 预训练权重
为方便你完成作业，我们提供了 SimCLR 模型的预训练权重（在 CIFAR-10 上训练约 18 小时）。运行下面的单元下载稍后会用到的预训练模型权重。（大约需要 1 分钟）""",
    ("assignment3/Self_Supervised_Learning.ipynb", 9): """# 数据增强

第一步是执行数据增强。在 `cs231n/simclr/data_utils.py` 中实现 `compute_train_transform()` 函数，应用下列随机变换：

1. 随机 resize 并裁剪到 `32x32`。
2. 以 0.5 的概率水平翻转图像。
3. 以 0.8 的概率应用 color jitter（定义见 `compute_train_transform()`）。
4. 以 0.2 的概率把图像转换为灰度。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 10): """现在完成 `cs231n/simclr/data_utils.py` 中的 `compute_train_transform()` 和 `CIFAR10Pair.__getitem__()`，应用数据增强变换，并生成 **$\\tilde{x}_i$** 和 **$\\tilde{x}_j$**。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 11): """测试并确认你的数据增强代码正确：""",
    ("assignment3/Self_Supervised_Learning.ipynb", 14): """# Base Encoder 和 Projection Head
接下来，把 base encoder 和 projection head 应用于增强样本 **$\\tilde{x}_i$** 和 **$\\tilde{x}_j$**。

base encoder $f$ 会为增强样本提取表示向量。SimCLR 论文发现，使用更深、更宽的模型可以提升性能，因此选择 [ResNet](https://arxiv.org/pdf/1512.03385.pdf) 作为 base encoder。base encoder 的输出是表示向量 **$h_i = f(\\tilde{x}_i$)** 和 **$h_j = f(\\tilde{x}_j$)**。

projection head $g$ 是一个小型神经网络，它把表示向量 **$h_i$** 和 **$h_j$** 映射到应用对比损失的空间。论文发现，使用非线性 projection head 可以提升其前一层的表示质量。具体来说，他们使用一个含一层隐藏层的 MLP 作为 projection head $g$。对比损失随后基于输出 **$z_i = g(h_i$)** 和 **$z_j = g(h_j$)** 计算。

我们在 `cs231n/simclr/model.py` 中提供了这两部分的实现。请快速阅读该文件，确保你理解实现。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 15): """# SimCLR：对比损失

一个包含 $N$ 张训练图像的 mini-batch 会产生总共 $2N$ 个数据增强样本。对每个增强样本正样本对 $(i, j)$，对比损失函数的目标是最大化向量 $z_i$ 和 $z_j$ 的一致性。具体来说，该损失是 normalized temperature-scaled cross entropy loss，目标是相对于 batch 中所有其他增强样本，最大化 $z_i$ 和 $z_j$ 的一致性：""",
    ("assignment3/Self_Supervised_Learning.ipynb", 16): """$$
l \\; (i, j) = -\\log \\frac{\\exp (\\;\\text{sim}(z_i, z_j)\\; / \\;\\tau) }{\\sum_{k=1}^{2N} \\mathbb{1}_{k \\neq i} \\exp (\\;\\text{sim} (z_i, z_k) \\;/ \\;\\tau) }
$$""",
    ("assignment3/Self_Supervised_Learning.ipynb", 17): """其中 $\\mathbb{1} \\in \\{0, 1\\}$ 是指示函数，当 $k\\neq i$ 时输出 1，否则输出 0。$\\tau$ 是 temperature 参数，决定指数项增长速度。

sim$(z_i, z_j) = \\frac{z_i \\cdot z_j}{|| z_i || || z_j ||}$ 是向量 $z_i$ 和 $z_j$ 之间的归一化点积。$z_i$ 和 $z_j$ 越相似，点积越大，分子也越大。分母通过对 batch 中 $z_i$ 和所有其他增强样本 $k$ 求和来归一化该值。归一化值范围为 $(0, 1)$；接近 1 的高分表示正样本对 $(i, j)$ 相似度高，同时 $i$ 与 batch 中其他增强样本 $k$ 相似度低。取负对数后，会把 $(0, 1)$ 范围映射到损失值。

总损失在 batch 中所有正样本对 $(i, j)$ 上计算。令 $z = [z_1, z_2, ..., z_{2N}]$ 包含 batch 中所有增强样本，其中 $z_{1}...z_{N}$ 是左分支输出，$z_{N+1}...z_{2N}$ 是右分支输出。因此，正样本对为 $(z_{k}, z_{k + N})$，对所有 $k \\in [1, N]$ 成立。

于是总损失 $L$ 为：""",
    ("assignment3/Self_Supervised_Learning.ipynb", 18): """$$
L = \\frac{1}{2N} \\sum_{k=1}^N [ \\; l(k, \\;k+N) + l(k+N, \\;k)\\;]
$$""",
    ("assignment3/Self_Supervised_Learning.ipynb", 19): """**注意：** 这个公式与论文中的公式略有不同。我们重新排列了 batch 中正样本对的顺序，所以索引不同。这个重新排列会让向量化实现更容易。

我们会逐步实现向量化形式的损失函数。请在 `cs231n/simclr/contrastive_loss.py` 中实现 `sim` 和 `simclr_loss_naive` 函数。运行下面的 sanity check 测试你的代码。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 23): """现在实现向量化版本：在 `cs231n/simclr/contrastive_loss.py` 中实现 `sim_positive_pairs`、`compute_sim_matrix` 和 `simclr_loss_vectorized`。运行下面的 sanity check 测试你的代码。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 27): """# 实现 train 函数
完成 `cs231n/simclr/utils.py` 中的 `train()` 函数，获取模型输出，并使用 `simclr_loss_vectorized` 计算损失。（请查看 `cs231n/simclr/model.py` 中的 `Model` 类，理解模型流水线和返回值。）""",
    ("assignment3/Self_Supervised_Learning.ipynb", 29): """### 训练 SimCLR 模型

运行下面的单元加载预训练权重，并继续训练一小段时间。这部分大约需要 10 分钟，并会输出到 `pretrained_model/trained_simclr_model.pth`。

**注意：** 如果看到类似 `_[WARN] Cannot find rule for ..._` 的日志，不用担心。这些与 notebook 中使用的另一个模块有关。你可以通过我们提供的提示和注释验证代码改动是否正确。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 31): """# 微调线性层做分类

现在是时候检验表示向量了。

我们从 SimCLR 模型中移除 projection head，并接上一个线性层，用于微调一个简单分类任务。线性层之前的所有层都被冻结，只训练最终线性层的权重。我们会比较 SimCLR + finetuning 模型和 baseline 模型的表现；baseline 没有提前做自监督学习，模型中所有权重都会被训练。你将亲自看到自监督学习的力量，以及学到的表示向量如何提升下游任务性能。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 32): """## Baseline：不使用自监督学习
首先看看 baseline 模型。我们从 SimCLR 模型中移除 projection head，并接上一个线性层，用于微调简单分类任务。这里没有提前做自监督学习，模型中所有权重都会被训练。运行下面的单元。

**注意：** 如果看到较低但合理的性能，不用担心。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 35): """## 使用自监督学习

现在看看自监督学习能带来多少提升。这里我们使用你写的 simclr loss 预训练 SimCLR 模型，从模型中移除 projection head，然后使用线性层微调简单分类任务。""",
    ("assignment3/Self_Supervised_Learning.ipynb", 37): """### 绘制对比

绘制 baseline 模型（无预训练）和同一模型经过自监督学习预训练后的测试准确率对比。""",

    ("assignment3/Transformer_Captioning.ipynb", 0): AI_USE,
    ("assignment3/Transformer_Captioning.ipynb", 1): STUDENT_DECLARATION,
    ("assignment3/Transformer_Captioning.ipynb", 3): """# 使用 Transformer 做图像描述
你已经为图像描述任务实现了 vanilla RNN。在这个 notebook 中，你将实现 transformer decoder 的关键部分，以完成同一任务。

**注意：** 与 RNN notebook 不同，本 notebook 主要使用 PyTorch，而不是 NumPy。""",
    ("assignment3/Transformer_Captioning.ipynb", 5): """# COCO 数据集
和前面的 notebook 一样，我们将使用 COCO 数据集做图像描述。""",
    ("assignment3/Transformer_Captioning.ipynb", 7): """# Transformer
你已经看到，RNN 非常强大，但通常训练较慢。此外，RNN 很难编码长距离依赖（LSTM 是缓解这个问题的一种方式）。2017 年，Vaswani 等人在论文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 中提出 Transformer，目的是引入并行性，并让模型学习长距离依赖。这篇论文不仅催生了自然语言处理领域中著名的 BERT 和 GPT 等模型，也引发了包括视觉在内的多个领域的广泛兴趣。这里我们在图像描述背景下介绍该模型，但 attention 思想本身更加通用。""",
    ("assignment3/Transformer_Captioning.ipynb", 8): """# Transformer：Multi-Headed Attention

### Dot-Product Attention

回忆一下，attention 可以看作作用在一个 query $q\\in\\mathbb{R}^d$、一组 value vectors $\\{v_1,\\dots,v_n\\}, v_i\\in\\mathbb{R}^d$，以及一组 key vectors $\\{k_1,\\dots,k_n\\}, k_i \\in \\mathbb{R}^d$ 上的操作，定义为：""",
    ("assignment3/Transformer_Captioning.ipynb", 9): """\\begin{align}
c &= \\sum_{i=1}^{n} v_i \\alpha_i \\\\
\\alpha_i &= \\frac{\\exp(k_i^\\top q)}{\\sum_{j=1}^{n} \\exp(k_j^\\top q)} \\\\
\\end{align}""",
    ("assignment3/Transformer_Captioning.ipynb", 10): """其中 $\\alpha_i$ 通常称为 “attention weights”，输出 $c\\in\\mathbb{R}^d$ 是对 value vectors 的加权平均。

### Self-Attention
在 Transformer 中，我们执行 self-attention，也就是说 values、keys 和 query 都由输入 $X \\in \\mathbb{R}^{\\ell \\times d}$ 派生，其中 $\\ell$ 是序列长度。具体来说，我们学习参数矩阵 $V,K,Q \\in \\mathbb{R}^{d\\times d}$，把输入 $X$ 映射如下：""",
    ("assignment3/Transformer_Captioning.ipynb", 11): """\\begin{align}
v_i = Vx_i\\ \\ i \\in \\{1,\\dots,\\ell\\}\\\\
k_i = Kx_i\\ \\ i \\in \\{1,\\dots,\\ell\\}\\\\
q_i = Qx_i\\ \\ i \\in \\{1,\\dots,\\ell\\}
\\end{align}""",
    ("assignment3/Transformer_Captioning.ipynb", 12): """### Multi-Headed Scaled Dot-Product Attention
在 multi-headed attention 中，我们为每个 head 学习一个参数矩阵，这让模型有更强表达能力，可以关注输入的不同部分。令 $h$ 为 head 数量，$Y_i$ 为第 $i$ 个 head 的 attention 输出。因此，我们学习单独的矩阵 $Q_i$、$K_i$ 和 $V_i$。为了让整体计算量与 single-headed 情况相同，我们选择 $Q_i \\in \\mathbb{R}^{d\\times d/h}$、$K_i \\in \\mathbb{R}^{d\\times d/h}$ 和 $V_i \\in \\mathbb{R}^{d\\times d/h}$。在上面的简单 dot-product attention 中加入缩放项 $\\frac{1}{\\sqrt{d/h}}$ 后，有：""",
    ("assignment3/Transformer_Captioning.ipynb", 13): """\\begin{equation} \\label{qkv_eqn}
Y_i = \\text{softmax}\\bigg(\\frac{(XQ_i)(XK_i)^\\top}{\\sqrt{d/h}}\\bigg)(XV_i)
\\end{equation}""",
    ("assignment3/Transformer_Captioning.ipynb", 14): """其中 $Y_i\\in\\mathbb{R}^{\\ell \\times d/h}$，$\\ell$ 是序列长度。

在我们的实现中，会对 attention weights 应用 dropout（尽管实践中 dropout 也可以用于其他步骤）：""",
    ("assignment3/Transformer_Captioning.ipynb", 15): """\\begin{equation} \\label{qkvdropout_eqn}
Y_i = \\text{dropout}\\bigg(\\text{softmax}\\bigg(\\frac{(XQ_i)(XK_i)^\\top}{\\sqrt{d/h}}\\bigg)\\bigg)(XV_i)
\\end{equation}""",
    ("assignment3/Transformer_Captioning.ipynb", 16): """最后，self-attention 的输出是所有 head 拼接结果的线性变换：""",
    ("assignment3/Transformer_Captioning.ipynb", 17): """\\begin{equation}
Y = [Y_1;\\dots;Y_h]A
\\end{equation}""",
    ("assignment3/Transformer_Captioning.ipynb", 18): """其中 $A \\in\\mathbb{R}^{d\\times d}$，且 $[Y_1;\\dots;Y_h]\\in\\mathbb{R}^{\\ell \\times d}$。

在 `cs231n/transformer_layers.py` 文件的 `MultiHeadAttention` 类中实现 multi-headed scaled dot-product attention。下面的代码会检查你的实现。相对误差应小于 `7e-3`。""",
    ("assignment3/Transformer_Captioning.ipynb", 20): """# Positional Encoding

虽然 transformer 可以轻松关注输入中的任意部分，但 attention 机制本身没有 token 顺序概念。然而，对许多任务（尤其是自然语言处理）来说，相对 token 顺序非常重要。为恢复顺序信息，作者向每个 word token 的 embedding 中加入 positional encoding。

令矩阵 $P \\in \\mathbb{R}^{l\\times d}$，其中 $P_{ij} = $""",
    ("assignment3/Transformer_Captioning.ipynb", 21): """$$
\\begin{cases}
\\text{sin}\\left(i \\cdot 10000^{-\\frac{j}{d}}\\right) & \\text{if j is even} \\\\
\\text{cos}\\left(i \\cdot 10000^{-\\frac{(j-1)}{d}}\\right) & \\text{otherwise} \\\\
\\end{cases}
$$""",
    ("assignment3/Transformer_Captioning.ipynb", 22): """我们不会把输入 $X \\in \\mathbb{R}^{l\\times d}$ 直接传入网络，而是传入 $X + P$。

在 `cs231n/transformer_layers.py` 中的 `PositionalEncoding` 里实现这一层。完成后，运行下面的代码做一个简单测试。你应该看到 `1e-3` 数量级或更小的误差。""",
    ("assignment3/Transformer_Captioning.ipynb", 24): """# 内联问题 1

在设计上面介绍的 scaled dot product attention 时，有几个关键设计决策。解释为什么下列选择是有益的：
1. 使用多个 attention heads，而不是一个。
2. 在应用 softmax 函数之前除以 $\\sqrt{d/h}$。回忆一下，$d$ 是特征维度，$h$ 是 head 数量。
3. 在 attention 操作输出后添加一个线性变换。

每个选择只需要一两句话，但请具体说明：如果没有该实现细节会发生什么，为什么这种情况不理想，以及所提出的实现如何改善它。

**你的回答：**""",
    ("assignment3/Transformer_Captioning.ipynb", 25): """# Transformer Decoder Block

Transformer decoder layer 包含三个模块：（1）self attention，用于处理输入向量序列；（2）cross attention，用于基于可用上下文处理输入（在这里是图像特征）；（3）feedforward module，用于独立处理序列中的每个向量。完成 `cs231n/transformer_layers.py` 中 `TransformerDecoderLayer` 的实现，并在下面测试。相对误差应小于 `1e-6`。

Transformer decoder layer 的三个主要组件是：（1）处理输入向量序列的 self-attention 模块，（2）融合额外上下文的 cross-attention 模块（在这里是图像特征），以及（3）独立处理序列中每个向量的 feedforward 模块。完成 `cs231n/transformer_layers.py` 中 `TransformerDecoderLayer` 的实现，并在下面测试。相对误差应小于 `1e-6`。""",
    ("assignment3/Transformer_Captioning.ipynb", 27): """# 用 Transformer 做图像描述
现在你已经实现了前面的层，可以把它们组合成基于 Transformer 的图像描述模型。打开 `cs231n/classifiers/transformer.py` 文件，查看 `CaptioningTransformer` 类。

实现该类的 `forward` 函数。完成后，运行下面的代码，用一个小测试用例检查前向传播；你应该看到 `1e-5` 数量级或更小的误差。""",
    ("assignment3/Transformer_Captioning.ipynb", 29): """# 在小数据上过拟合 Transformer 图像描述模型
运行下面的代码，在与之前 RNN 相同的小数据集上，让基于 Transformer 的图像描述模型过拟合。""",
    ("assignment3/Transformer_Captioning.ipynb", 31): """打印最终训练损失。你应该看到最终损失小于 0.05。""",
    ("assignment3/Transformer_Captioning.ipynb", 33): """# 测试时的 Transformer 采样
采样代码已经为你写好。你只需运行下面的代码，把结果与之前的 RNN 结果进行比较。和之前一样，由于训练数据很少，训练集上的结果应该明显好于验证集结果。""",
    ("assignment3/Transformer_Captioning.ipynb", 35): """# Vision Transformer（ViT）

[Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929) 表明，把 transformer 模型应用到图像 patch 序列上（称为 Vision Transformer）不仅能取得令人印象深刻的性能，而且在大数据集上训练时，相比卷积神经网络能更有效地扩展。我们将使用已有 transformer 组件实现一个 Vision Transformer 版本，并在 CIFAR-10 数据集上训练它。""",
    ("assignment3/Transformer_Captioning.ipynb", 36): """Vision Transformer 会把输入图像转换为固定大小的 patch 序列，并把每个 patch embedding 成一个潜在向量。在 `cs231n/transformer_layers.py` 中完成 `PatchEmbedding` 的实现，并在下面测试。你应该看到小于 `1e-4` 的相对误差。""",
    ("assignment3/Transformer_Captioning.ipynb", 38): """patch 向量序列会由 transformer encoder layers 处理，每层包含 self-attention 和 feed-forward 模块。由于所有向量都会相互 attend，attention masking 并非严格必要。不过，为保持一致性，我们仍然实现它。

在 `cs231n/transformer_layers.py` 中实现 `TransformerEncoderLayer`，并在下面测试。你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/Transformer_Captioning.ipynb", 40): """查看 `cs231n/classifiers/transformer.py` 中的 `VisionTransformer` 实现。

对于分类任务，ViT 会把输入图像划分为 patch，并用 transformer 处理 patch 向量序列。最后，对所有 patch 向量做 average pooling，并用于预测图像类别。我们会使用同样的一维 sinusoidal positional encoding 注入顺序信息，不过二维 sinusoidal positional encoding 和可学习 positional encoding 也都是有效选择。

完成 ViT 前向传播并在下面测试。你应该看到小于 `1e-6` 的相对误差。""",
    ("assignment3/Transformer_Captioning.ipynb", 42): """我们首先通过让模型在一个训练 batch 上过拟合来验证实现。请相应调节 learning rate 和 weight decay。""",
    ("assignment3/Transformer_Captioning.ipynb", 46): """现在我们将在整个数据集上训练它。""",
    ("assignment3/Transformer_Captioning.ipynb", 49): """# 内联问题 2

尽管 ViT 最近在大规模图像识别任务中很成功，但在较小数据集上训练时，它们往往落后于传统 CNN。造成这种性能差距的底层因素是什么？可以使用哪些技术提升 ViT 在小数据集上的性能？

**你的回答**：在此填写。""",
    ("assignment3/Transformer_Captioning.ipynb", 50): """# 内联问题 3

如果分别做下列改变，ViT 中 self-attention 层的计算成本会如何变化？请忽略 QKV 和输出投影的计算成本。

(i) hidden dimension 翻倍。
(ii) 输入图像的高度和宽度都翻倍。
(iii) patch size 翻倍。
(iv) layer 数量翻倍。

**你的回答**：在此填写。""",
    ("assignment3/collect_submission.ipynb", 1): """# 收集提交文件：压缩代码并生成 PDF

<font color="red">**注意：** 运行这一步之前，请记得执行所有 notebook 的全部单元。缺失的单元输出不会被重新评分。</font>

完成所有其他 notebook 后再运行这个 notebook。

它会：

* 生成一个包含你的代码（`.py` 和 `.ipynb`）的压缩包，文件名为 `a3_code_submission.zip`。
* 将所有 notebook 转换成一个 PDF 文件，文件名为 `a3_inline_submission.pdf`。

如果这一步提交文件生成成功，你应该会看到如下显示信息：

`### Done! Please submit a3_code_submission.zip and a3_inline_submission.pdf to Gradescope. ###`

请确保把 zip 和 pdf 文件下载到本地电脑，然后提交到 Gradescope。恭喜你完成本次作业。""",
})


def main() -> None:
    for rel_path in sorted({key[0] for key in TRANSLATIONS}):
        path = ROOT / rel_path
        nb = nbformat.read(path, as_version=4)
        for (candidate, cell_index), text in TRANSLATIONS.items():
            if candidate == rel_path:
                nb.cells[cell_index].source = text
        nb.nbformat_minor = max(int(getattr(nb, "nbformat_minor", 0)), 5)
        _, nb = normalize(nb)
        nbformat.validate(nb)
        nbformat.write(nb, path)


if __name__ == "__main__":
    main()
