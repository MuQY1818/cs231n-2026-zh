from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


COMMENT_EXACT = {
    "# TODO:": "# TODO：",
    "# TODO": "# TODO",
    "# Setup cell.": "# 设置单元。",
    "# Set default size of plots.": "# 设置图像的默认大小。",
    "# set default size of plots": "# 设置图像的默认大小",
    "# Experiment with this!": "# 可以尝试修改这里。",
    "# Experiment with this": "# 可以尝试修改这里",
    "# Do not modify this cell.": "# 不要修改这个单元。",
    "# Load the raw CIFAR-10 data": "# 加载原始 CIFAR-10 数据",
    "# Load the raw CIFAR-10 data.": "# 加载原始 CIFAR-10 数据。",
    "# Load the (preprocessed) CIFAR-10 data.": "# 加载预处理后的 CIFAR-10 数据。",
    "# Load the (preprocessed) CIFAR10 data.": "# 加载预处理后的 CIFAR-10 数据。",
    "# Load COCO data from disk into a dictionary.": "# 从磁盘加载 COCO 数据到字典中。",
    "# Print out results.": "# 打印结果。",
    "# Print out all the keys and values from the data dictionary.": "# 打印数据字典中的所有键和值。",
    "# Print final training accuracy.": "# 打印最终训练准确率。",
    "# Print final validation accuracy.": "# 打印最终验证准确率。",
    "# Print final training loss.": "# 打印最终训练损失。",
    "# Save best model": "# 保存最佳模型",
    "# Save best softmax model": "# 保存最佳 softmax 模型",
    "# Save the trained model for autograder.": "# 保存训练好的模型，供 autograder 使用。",
    "# Run some setup code for this notebook.": "# 运行本 notebook 的初始化代码。",
    "# Load the raw CIFAR-10 data.": "# 加载原始 CIFAR-10 数据。",
    "# Subsample the data": "# 对数据做子采样",
    "# Normalize the data: subtract the mean image": "# 归一化数据：减去均值图像",
    "# Transpose so that channels come first": "# 转置数据，使通道维度在前",
    "# Package data into a dictionary": "# 将数据打包到字典中",
    "# Preprocessing: subtract the mean image": "# 预处理：减去均值图像",
    "# Preprocessing: Add a bias dimension": "# 预处理：添加 bias 维度",
    "# Preprocessing: Remove the bias dimension": "# 预处理：移除 bias 维度",
    "# Preprocessing: reshape the image data into rows": "# 预处理：把图像数据 reshape 成行向量",
    "# first: compute the image mean based on the training data": "# 第一步：基于训练数据计算图像均值",
    "# visualize the mean image": "# 可视化均值图像",
    "# second: subtract the mean image from train and test data": "# 第二步：从训练和测试数据中减去均值图像",
    "# normalize to [0,1]": "# 归一化到 [0,1]",
    "# evaluate loss and gradient": "# 计算损失和梯度",
    "# perform parameter update": "# 执行参数更新",
    "# Compute loss and gradient": "# 计算损失和梯度",
    "# Perform a parameter update": "# 执行参数更新",
    "# Maybe print training loss": "# 按需打印训练损失",
    "# Maybe subsample the data": "# 按需对数据做子采样",
    "# Maybe subsample the training data": "# 按需对训练数据做子采样",
    "# Make a minibatch of training data": "# 构造一个训练数据 minibatch",
    "# Compute predictions in batches": "# 分 batch 计算预测结果",
    "# Keep track of the best model": "# 跟踪最佳模型",
    "# At the end of training swap the best params into the model": "# 训练结束时，把最佳参数写回模型",
    "# At the end of every epoch, increment the epoch counter.": "# 每个 epoch 结束时，递增 epoch 计数器。",
    "# Unpack keyword arguments": "# 解包关键字参数",
    "# Throw an error if there are extra keyword arguments": "# 如果存在额外关键字参数，则抛出错误",
    "# Set up some variables for book-keeping": "# 设置若干变量用于记录训练过程",
    "# Cast all parameters to the correct datatype.": "# 将所有参数转换为正确的数据类型。",
    "# If test mode return early.": "# 如果是测试模式，则提前返回。",
    "# Store the updated running means back into bn_param": "# 将更新后的 running mean 写回 bn_param",
    "# Initialize the loss and gradient to zero.": "# 将损失和梯度初始化为零。",
    "# compute the loss and the gradient": "# 计算损失和梯度",
    "# compute the probabilities in numerically stable way": "# 以数值稳定的方式计算概率",
    "# normalize": "# 归一化",
    "# negative log probability is the loss": "# 负对数概率即为损失",
    "# normalized hinge loss plus regularization": "# 归一化 hinge loss 加正则化",
    "# return histogram": "# 返回直方图",
    "# convert rgb to grayscale if needed": "# 如果需要，将 RGB 转为灰度",
    "# image size": "# 图像大小",
    "# number of gradient bins": "# 梯度 bin 数量",
    "# pixels per cell": "# 每个 cell 的像素数",
    "# compute gradient on x-direction": "# 计算 x 方向梯度",
    "# compute gradient on y-direction": "# 计算 y 方向梯度",
    "# gradient magnitude": "# 梯度幅值",
    "# gradient orientation": "# 梯度方向",
    "# number of cells in x": "# x 方向 cell 数量",
    "# number of cells in y": "# y 方向 cell 数量",
    "# compute orientations integral images": "# 计算方向积分图",
    "# create new integral image for this orientation": "# 为该方向创建新的积分图",
    "# isolate orientations in this range": "# 筛选该范围内的方向",
    "# select magnitudes for those orientations": "# 选择这些方向对应的幅值",
    "# grayscale image": "# 灰度图像",
    "# rgb image": "# RGB 图像",
    "# loop through three color channels": "# 遍历三个颜色通道",
    "# compute normalized histogram": "# 计算归一化直方图",
    "# concatenate histogram": "# 拼接直方图",
    "# unknown image type": "# 未知图像类型",
    "# evaluate function value at original point": "# 在原始点计算函数值",
    "# iterate over all indexes in x": "# 遍历 x 中的所有索引",
    "# evaluate function at x+h": "# 在 x+h 处计算函数值",
    "# evalute f(x + h)": "# 计算 f(x + h)",
    "# evaluate f(x - h)": "# 计算 f(x - h)",
    "# restore": "# 恢复原值",
    "# reset": "# 重置",
    "# increment by h": "# 增加 h",
    "# compute the partial derivative with centered formula": "# 使用中心差分公式计算偏导数",
    "# the slope": "# 斜率",
    "# step to next dimension": "# 前进到下一个维度",
    "# Create a placeholder, to be overwritten by your code below.": "# 创建占位变量，后续由你的代码覆盖。",
    "# Create output": "# 创建输出",
    "# Pad the input": "# 对输入做 padding",
    "# Zero-pad the input": "# 对输入做零填充",
    "# Figure out output dimensions": "# 计算输出维度",
    "# Reshape the output": "# reshape 输出",
    "# Be nice and return a contiguous array": "# 返回连续数组，便于后续处理",
    "# Compute and display the accuracy": "# 计算并显示准确率",
    "# Visualize the learned weights for each class.": "# 可视化每个类别学到的权重。",
    "# strip out the bias": "# 去掉 bias",
    "# Rescale the weights to be between 0 and 255": "# 将权重缩放到 0 到 255 之间",
    "# Set up a convolutional weights holding 2 filters, each 3x3": "# 设置包含 2 个 3x3 滤波器的卷积权重",
    "# The first filter converts the image to grayscale.": "# 第一个滤波器将图像转换为灰度。",
    "# Set up the red, green, and blue channels of the filter.": "# 设置滤波器的红、绿、蓝通道。",
    "# Second filter detects horizontal edges in the blue channel.": "# 第二个滤波器检测蓝色通道中的水平边缘。",
    "# Vector of biases. We don't need any bias for the grayscale": "# bias 向量。灰度滤波器不需要 bias",
    "# Show the original images and the results of the conv operation": "# 显示原始图像和卷积操作结果",
    "# Remember to restart the runtime after executing this cell!": "# 执行这个单元后，请记得重启 runtime。",
    "# We will be using float throughout this tutorial.": "# 本教程全程使用 float。",
    "# Constant to control how frequently we print train loss.": "# 控制训练损失打印频率的常量。",
    "# read in N, C, H, W": "# 读入 N、C、H、W",
    "# useful stateless functions": "# 有用的无状态函数",
    "# first we flatten the image": "# 首先 flatten 图像",
    "# Move the data to the proper device (GPU or CPU)": "# 将数据移动到合适设备（GPU 或 CPU）",
    "# Forward pass: compute scores and loss": "# 前向传播：计算 scores 和 loss",
    "# Manually zero the gradients after running the backward pass": "# 运行反向传播后手动清零梯度",
    "# assign layer objects to class attributes": "# 将层对象赋给类属性",
    "# forward always defines connectivity": "# forward 始终定义连接关系",
    "# set model to evaluation mode": "# 将模型设置为评估模式",
    "# move the model parameters to CPU/GPU": "# 将模型参数移动到 CPU/GPU",
    "# put model to training mode": "# 将模型设置为训练模式",
    "# Actually update the parameters of the model using the gradients": "# 使用梯度实际更新模型参数",
    "# We need to wrap `flatten` function in a module in order to stack it": "# 为了把 `flatten` 堆叠到 Sequential 中，需要将其包装为 module",
    "# you can use Nesterov momentum in optim.SGD": "# 可以在 optim.SGD 中使用 Nesterov momentum",
    "# Initialize word vectors": "# 初始化词向量",
    "# Initialize CNN -> hidden state projection parameters": "# 初始化 CNN 到 hidden state 的投影参数",
    "# Initialize parameters for the RNN": "# 初始化 RNN 参数",
    "# Initialize output to vocab weights": "# 初始化 hidden 到词表输出的权重",
    "# Cast parameters to correct dtype": "# 将参数转换为正确 dtype",
    "# Unpack parameters": "# 解包参数",
    "# A helper function to register some constants as buffers to ensure that": "# 辅助函数：把一些常量注册为 buffer，以确保",
    "# they are on the same device as model parameters.": "# 它们与模型参数位于同一设备。",
    "# Each buffer can be accessed as `self.name`": "# 每个 buffer 可通过 `self.name` 访问",
    "# Noise schedule beta and alpha values": "# noise schedule 中的 beta 和 alpha 值",
    "# loss weight": "# 损失权重",
    "# We have already implemented this method for you.": "# 我们已经为你实现了这个方法。",
    "# Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1": "# 应用 color jitter 的变换：brightness=0.4、contrast=0.4、saturation=0.4、hue=0.1",
    "# encoder": "# encoder",
    "# projection head": "# projection head",
    "# generate feature bank": "# 生成 feature bank",
    "# loop test data to predict the label by weighted knn search": "# 遍历测试数据，用加权 kNN 搜索预测标签",
    "# counts for each class": "# 每个类别的计数",
    "# Embed time step": "# 嵌入时间步",
    "# Embed condition and add to context": "# 嵌入条件并加入 context",
    "# Randomly drop condition": "# 随机丢弃条件",
    "# Initial convolution": "# 初始卷积",
    "# Final convolution to map to the output channels": "# 最终卷积，用于映射到输出通道",
    "# Middle blocks": "# 中间 block",
    "# Downsampling blocks": "# 下采样 block",
    "# Upsampling blocks": "# 上采样 block",
    "# This mounts your Google Drive to the Colab VM.": "# 将你的 Google Drive 挂载到 Colab VM。",
    "# Now that we've mounted your Drive, this ensures that": "# 现在已经挂载 Drive，下面确保",
    "# the Python interpreter of the Colab VM can load": "# Colab VM 的 Python 解释器可以加载",
    "# python files from within it.": "# 其中的 Python 文件。",
    "# This downloads the CIFAR-10 dataset to your Drive": "# 将 CIFAR-10 数据集下载到你的 Drive",
    "# This downloads the COCO dataset to your Drive": "# 将 COCO 数据集下载到你的 Drive",
    "# if it doesn't already exist.": "# 如果它还不存在。",
    "# assignment folder, e.g. 'cs231n/assignments/assignment1/'": "# 作业文件夹，例如 'cs231n/assignments/assignment1/'",
    "# assignment folder, e.g. 'cs231n/assignments/assignment2/'": "# 作业文件夹，例如 'cs231n/assignments/assignment2/'",
    "# assignment folder, e.g. 'cs231n/assignments/assignment3/'": "# 作业文件夹，例如 'cs231n/assignments/assignment3/'",
    "# e.g. 'cs231n/assignments/assignment1/'": "# 例如 'cs231n/assignments/assignment1/'",
    "# TODO: Enter the foldername in your Drive where you have saved the unzipped": "# TODO：填写 Drive 中保存解压后",
    "# TODO: Enter the path in your Drive of the assignment.": "# TODO：填写 Drive 中该作业的路径。",
    "# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)": "# 清理变量，避免多次加载数据导致内存问题",
    "# for auto-reloading extenrnal modules": "# 用于自动重新加载外部模块",
    "# for auto-reloading external modules": "# 用于自动重新加载外部模块",
    "# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython": "# 参考 http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython",
    "# This is a bit of magic to make matplotlib figures appear inline in the notebook": "# 这是一点 IPython magic，让 matplotlib 图像内嵌显示在 notebook 中",
    "# rather than in a new window.": "# 而不是显示在新窗口中。",
    "# Some more magic so that the notebook will reload external python modules;": "# 另一点 magic，用于让 notebook 重新加载外部 Python 模块；",
    "# As a sanity check, we print out the size of the training and test data.": "# 作为合理性检查，打印训练数据和测试数据的大小。",
    "# Visualize some examples from the dataset.": "# 可视化数据集中的一些样本。",
    "# We show a few examples of training images from each class.": "# 展示每个类别中的若干训练图像样本。",
    "# training set.": "# 训练集。",
    "# move to device, e.g. GPU": "# 移动到设备，例如 GPU",
    "# [B, K]": "# [B, K]",
    "# [2*N, D]": "# [2*N, D]",
    "# B x 3 x H x W": "# B x 3 x H x W",
    "# T x H x W x 3": "# T x H x W x 3",
    "# (b,)": "# (b,)",
}


PHRASES = [
    ("END OF YOUR CODE", "你的代码结束"),
    ("START OF YOUR CODE", "你的代码开始"),
    ("DO NOT DELETE/MODIFY THIS LINE", "不要删除或修改这一行"),
    ("TODO: Start of your code.", "TODO：你的代码从这里开始。"),
    ("TODO: Copy over your solution from Assignment 1.", "TODO：复制你在 Assignment 1 中的解答。"),
    ("TODO: Copy over your solution from A1.", "TODO：复制你在 A1 中的解答。"),
    ("TODO: Enter the foldername", "TODO：填写文件夹名"),
    ("TODO: Enter the path", "TODO：填写路径"),
    ("TODO: Train", "TODO：训练"),
    ("TODO: Use", "TODO：使用"),
    ("TODO: Set up", "TODO：设置"),
    ("TODO: Rewrite", "TODO：重写"),
    ("TODO: Instantiate", "TODO：实例化"),
    ("TODO: Initialize", "TODO：初始化"),
    ("TODO: Complete", "TODO：完成"),
    ("TODO: Apply", "TODO：应用"),
    ("TODO: Process", "TODO：处理"),
    ("TODO: Divide", "TODO：划分"),
    ("TODO: Construct", "TODO：构造"),
    ("TODO: Index into", "TODO：索引到"),
    ("TODO: Implement", "TODO：实现"),
    ("Start of your code", "你的代码开始"),
    ("End of your code", "你的代码结束"),
    ("END OF YOUR CODE", "你的代码结束"),
    ("START OF YOUR CODE", "你的代码开始"),
    ("NOTE:", "注意："),
    ("HINT:", "提示："),
    ("Hint:", "提示："),
    ("Note:", "注意："),
    ("Store the result", "将结果存储"),
    ("Store the data", "将数据存储"),
    ("Store these labels", "将这些标签存储"),
    ("Store this label", "将该标签存储"),
    ("Store the updated", "将更新后的"),
    ("Store the dropout mask", "将 dropout mask 存储"),
    ("storing the result", "将结果存储"),
    ("storing the next value", "将下一个值存储"),
    ("storing the loss and gradients", "存储损失和梯度"),
    ("Store your best model", "将你的最佳模型存储"),
    ("Store your best trained", "存储你训练得到的最佳"),
    ("Store weights and biases", "存储权重和偏置"),
    ("storing all values", "存储所有值"),
    ("You should", "你应该"),
    ("you should", "你应该"),
    ("You will need", "你需要"),
    ("you will need", "你需要"),
    ("You can", "你可以"),
    ("you can", "你可以"),
    ("You may", "你可以"),
    ("you may", "你可以"),
    ("we need", "我们需要"),
    ("We need", "我们需要"),
    ("we have", "我们已经"),
    ("We have", "我们已经"),
    ("we will", "我们将"),
    ("We will", "我们将"),
    ("we use", "我们使用"),
    ("We use", "我们使用"),
    ("we set", "我们设置"),
    ("We set", "我们设置"),
    ("we wrap", "我们包装"),
    ("We wrap", "我们包装"),
    ("Don't forget", "不要忘记"),
    ("Do not", "不要"),
    ("do not", "不要"),
    ("not use", "不要使用"),
    ("nor use", "也不要使用"),
    ("without using", "不使用"),
    ("no loops are allowed", "不允许使用循环"),
    ("For simplicity", "为简单起见"),
    ("For each", "对于每个"),
    ("For now", "目前"),
    ("During training", "训练时"),
    ("During testing", "测试时"),
    ("test-time", "测试时"),
    ("training-time", "训练时"),
    ("train/test", "训练/测试"),
    ("forward pass", "前向传播"),
    ("backward pass", "反向传播"),
    ("forward", "前向"),
    ("backward", "反向"),
    ("upstream", "上游"),
    ("gradient", "梯度"),
    ("gradients", "梯度"),
    ("weights", "权重"),
    ("biases", "偏置"),
    ("loss", "损失"),
    ("scores", "分数"),
    ("input", "输入"),
    ("output", "输出"),
    ("outputs", "输出"),
    ("samples", "样本"),
    ("examples", "样本"),
    ("training data", "训练数据"),
    ("validation", "验证"),
    ("test set", "测试集"),
    ("training set", "训练集"),
    ("minibatch", "minibatch"),
    ("batch size", "batch size"),
    ("shape", "形状"),
    ("Initialize", "初始化"),
    ("initialize", "初始化"),
    ("Compute", "计算"),
    ("compute", "计算"),
    ("computing", "计算"),
    ("computed", "计算得到的"),
    ("using", "使用"),
    ("Use", "使用"),
    ("use", "使用"),
    ("according to", "根据"),
    ("according", "根据"),
    ("classification", "分类"),
    ("classifier", "分类器"),
    ("prediction", "预测"),
    ("predicted", "预测的"),
    ("parameters", "参数"),
    ("implementation", "实现"),
    ("autograder", "autograder"),
    ("automated tests", "自动测试"),
    ("reference output", "参考输出"),
    ("relative error", "相对误差"),
    ("relative errors", "相对误差"),
    ("less than", "小于"),
    ("should be", "应为"),
    ("should see", "应该看到"),
    ("should match", "应匹配"),
    ("close to", "接近"),
    ("around", "约为"),
    ("Returns", "返回"),
    ("Inputs", "输入"),
    ("Input", "输入"),
    ("Args", "参数"),
    ("Returns", "返回"),
    ("A tuple of", "一个 tuple，包含"),
    ("of shape", "形状为"),
    ("same shape as", "形状与其相同"),
    ("same shape", "相同形状"),
    ("Number of", "数量"),
    ("number of", "数量"),
    ("Dimension of", "维度"),
    ("dimension of", "维度"),
    ("hidden dimension", "隐藏维度"),
    ("feature dimension", "特征维度"),
    ("image size", "图像大小"),
    ("image features", "图像特征"),
    ("Image features", "图像特征"),
    ("caption", "caption"),
    ("captions", "captions"),
    ("Noisy image", "带噪图像"),
    ("Starting image", "起始图像"),
    ("Predicted noise", "预测噪声"),
    ("Time step", "时间步"),
    ("tensor", "tensor"),
    ("Tensor", "Tensor"),
    ("Normalized dot product", "归一化点积"),
    ("contrastive loss", "对比损失"),
    ("vectorized version", "向量化版本"),
    ("No loops are allowed", "不允许使用循环"),
    ("Forward pass", "前向传播"),
    ("Backward pass", "反向传播"),
    ("Utility functions", "工具函数"),
    ("viewing and processing images", "查看和处理图像"),
    ("Load and resize an image from disk", "从磁盘加载并 resize 图像"),
    ("Preprocess an image", "预处理图像"),
    ("Undo preprocessing", "撤销预处理"),
    ("U-Net", "U-Net"),
    ("Google Drive", "Google Drive"),
    ("Colab VM", "Colab VM"),
    ("Python interpreter", "Python 解释器"),
    ("download", "下载"),
    ("downloads", "下载"),
    ("dataset", "数据集"),
    ("datasets", "数据集"),
    ("already exist", "已经存在"),
    ("doesn't already exist", "尚不存在"),
    ("mounted your Drive", "挂载你的 Drive"),
    ("assignment folder", "作业文件夹"),
    ("unzipped", "解压后"),
    ("external python modules", "外部 Python 模块"),
    ("matplotlib figures", "matplotlib 图像"),
    ("appear inline", "内嵌显示"),
    ("new window", "新窗口"),
    ("memory issue", "内存问题"),
    ("sanity check", "合理性检查"),
    ("print out", "打印"),
    ("Visualize", "可视化"),
    ("visualize", "可视化"),
    ("some examples", "一些样本"),
    ("from the dataset", "来自数据集"),
    ("from each class", "来自每个类别"),
    ("of the dataset", "数据集的"),
    ("of the model", "模型的"),
    ("of the network", "网络的"),
    ("of the loss", "损失的"),
    ("of the input", "输入的"),
    ("of the output", "输出的"),
    ("from the training data", "来自训练数据"),
    ("for autograder", "供 autograder 使用"),
    ("for visualization", "用于可视化"),
    ("for classification", "用于分类"),
    ("for training", "用于训练"),
    ("for inference", "用于推理"),
    ("training loss", "训练损失"),
    ("training accuracy", "训练准确率"),
    ("validation accuracy", "验证准确率"),
    ("test accuracy", "测试准确率"),
    ("learning rate", "学习率"),
    ("regularization", "正则化"),
    ("regularization strength", "正则化强度"),
    ("weight_scale", "weight_scale"),
    ("weight decay", "weight decay"),
    ("hyperparameters", "超参数"),
    ("tune", "调优"),
    ("tuning", "调优"),
    ("overfit", "过拟合"),
    ("dropout probability", "dropout 概率"),
    ("running mean", "running mean"),
    ("running variance", "running variance"),
    ("standard deviation", "标准差"),
    ("sample mean", "样本均值"),
    ("sample variance", "样本方差"),
    ("L2 regularization", "L2 正则化"),
    ("factor of 0.5", "0.5 的因子"),
    ("simplify the expression", "简化表达式"),
    ("with replacement", "有放回"),
    ("without replacement", "无放回"),
    ("faster than", "快于"),
    ("training point", "训练点"),
    ("test point", "测试点"),
    ("nearest neighbors", "最近邻"),
    ("distance matrix", "距离矩阵"),
    ("distance", "距离"),
    ("labels", "标签"),
    ("label", "标签"),
    ("break ties", "打破平局"),
    ("smaller", "更小"),
    ("numpy.argsort", "numpy.argsort"),
    ("matrix multiplication", "矩阵乘法"),
    ("broadcast sums", "broadcast 求和"),
    ("basic array operations", "基础数组操作"),
    ("functions from scipy", "scipy 函数"),
    ("intermediate values", "中间值"),
    ("data augmentation", "数据增强"),
    ("pretrained weights", "预训练权重"),
    ("feature bank", "feature bank"),
    ("weighted knn search", "加权 kNN 搜索"),
    ("classifier-free guidance", "classifier-free guidance"),
    ("skip connections", "skip connections"),
    ("Downsample", "下采样"),
    ("Upsample", "上采样"),
    ("ResnetBlock", "ResnetBlock"),
    ("context", "context"),
    ("condition", "condition"),
]


CLEANUP_PHRASES = [
    ("计算d", "计算得到的"),
    ("梯度s", "梯度"),
    ("标签s", "标签"),
    ("参数s", "参数"),
    ("分数s", "分数"),
    ("类别s", "类别"),
    ("输出s", "输出"),
    ("输入s", "输入"),
    ("examples", "样本"),
    ("example", "样本"),
    ("intermediates", "中间量"),
    ("statistics", "统计量"),
    ("variance", "方差"),
    ("mean", "均值"),
    ("stds", "标准差"),
    ("std", "标准差"),
    ("data types", "数据类型"),
    ("data type", "数据类型"),
    ("data", "数据"),
    ("class 分数", "类别分数"),
    ("class scores", "类别分数"),
    ("testing point", "测试点"),
    ("training point", "训练点"),
    ("neighbors", "邻居"),
    ("neighbor", "邻居"),
    ("variables", "变量"),
    ("variable", "变量"),
    ("dictionary", "字典"),
    ("function", "函数"),
    ("features", "特征"),
    ("feature", "特征"),
    ("dimension", "维度"),
    ("dimensions", "维度"),
    ("matrix", "矩阵"),
    ("array", "数组"),
    ("arrays", "数组"),
    ("values", "值"),
    ("value", "值"),
    ("result", "结果"),
    ("results", "结果"),
    ("output", "输出"),
    ("input", "输入"),
    ("loss", "损失"),
    ("class", "类别"),
    ("labels", "标签"),
    ("label", "标签"),
    ("points", "点"),
    ("point", "点"),
    ("mode", "模式"),
    ("layer", "层"),
    ("layers", "层"),
    ("batch normalization", "batch normalization"),
    ("batch norm", "batch norm"),
    ("dropout", "dropout"),
    ("normalization", "归一化"),
    ("regularization", "正则化"),
    ("implementation", "实现"),
    ("automated tests", "自动测试"),
    ("make sure", "确保"),
    ("matches ours", "与参考实现匹配"),
    ("you pass", "能通过"),
    ("includes", "包含"),
    ("include", "包含"),
    ("factor", "因子"),
    ("simplify", "简化"),
    ("expression", "表达式"),
    ("incoming", "输入的"),
    ("normalized", "归一化后的"),
    ("scale", "缩放"),
    ("shift", "平移"),
    ("cache", "cache"),
    ("forward", "前向"),
    ("backward", "反向"),
    ("train", "训练"),
    ("test", "测试"),
    ("based on", "基于"),
    ("reference", "参考"),
    ("original paper", "原始论文"),
    ("prove to be helpful", "可能会有帮助"),
    ("might", "可能"),
    ("should", "应该"),
    ("will", "将"),
    ("can", "可以"),
    ("need", "需要"),
    ("look up", "查阅"),
    ("find", "找到"),
    ("store", "存储"),
    ("Store", "存储"),
    ("break ties", "打破平局"),
    ("choosing", "选择"),
    ("smaller", "更小的"),
    ("where", "其中"),
    ("with", "使用"),
    ("without", "不使用"),
    ("using", "使用"),
    ("from", "来自"),
    ("into", "到"),
    ("over", "在"),
    ("then", "然后"),
    ("these", "这些"),
    ("this", "这个"),
    ("that", "该"),
    (" for ", " 用于 "),
    (" and ", " 并 "),
    (" in ", " 在 "),
    (" of ", " 的 "),
    (" to ", " 到 "),
]


ARTIFACT_REPAIRS = [
    ("模式ls", "模型"),
    ("模式l", "模型"),
    ("类别es", "类别"),
    ("类别s", "类别"),
    ("层s", "层"),
    ("函数s", "函数"),
    ("距离s", "距离"),
    ("测试点s", "测试点"),
    ("训练点s", "训练点"),
    ("预测s", "预测"),
    ("损失es", "损失"),
    ("维度s", "维度"),
    ("均值s", "均值"),
    ("方差s", "方差"),
    ("结果s", "结果"),
    ("数组s", "数组"),
    ("值s", "值"),
    ("计算s", "计算"),
    ("使用d", "使用"),
    ("需要ed", "需要"),
    ("训练ing", "训练"),
    ("测试ing", "测试"),
    ("初始化d", "初始化"),
    ("存储d", "存储"),
    ("缩放d", "缩放"),
    ("平移ing", "平移"),
    ("反向s", "反向"),
    ("vali数据ion", "验证"),
    ("gray缩放", "灰度"),
    ("shor测试", "最短"),
    ("数据loader", "DataLoader"),
    ("Re形状", "reshape"),
    ("re形状", "reshape"),
    ("计算_距离", "compute_distances"),
    ("使用out", "不使用"),
    ("c所有ing", "调用"),
    ("c所有ed", "调用"),
    ("c所有", "调用"),
    ("所有ocate", "分配"),
    ("所有ows", "允许"),
    ("所有ow", "允许"),
    ("par所有el", "并行"),
    ("Specific所有y", "具体来说"),
    ("Specifi调用y", "具体来说"),
    ("Fin所有y", "最后"),
    ("Rec所有", "回忆"),
    ("manu所有y", "手动"),
    ("periodic所有y", "定期"),
    ("rec在", "恢复"),
    ("C所有able", "Callable"),
    ("stack在flow", "stackoverflow"),
    ("所有ow-pickle", "allow-pickle"),
    ("to re使用", "复用"),
    ("re使用", "复用"),
    ("to be 使用", "被使用"),
    ("到 be 使用", "被使用"),
    ("should not", "不应该"),
    ("isn't", "并不是"),
    ("aren't", "不是"),
]


DOC_REPLACEMENTS = [
    ("This file defines layer types that are commonly used for recurrent neural networks.", "本文件定义循环神经网络中常用的层类型。"),
    ("This file defines layer types that are commonly used for transformers.", "本文件定义 Transformer 中常用的层类型。"),
    ("Class for a multi-layer fully connected neural network.", "多层全连接神经网络类。"),
    ("Save model parameters.", "保存模型参数。"),
    ("Load model parameters.", "加载模型参数。"),
    ("Initialize a new FullyConnectedNet.", "初始化新的 FullyConnectedNet。"),
    ("Compute loss and gradient for the fully connected net.", "计算全连接网络的损失和梯度。"),
    ("A subclass that uses the Multiclass SVM loss function", "使用 Multiclass SVM 损失函数的子类"),
    ("A subclass that uses the Softmax + Cross-entropy loss function", "使用 Softmax + Cross-entropy 损失函数的子类"),
    ("Convert RGB image to grayscale", "将 RGB 图像转换为灰度图像"),
    ("Compute Histogram of Gradient (HOG) feature for an image", "计算图像的方向梯度直方图（HOG）特征"),
    ("Compute color histogram feature for an image", "计算图像的颜色直方图特征"),
    ("visualize a grid of images", "可视化图像网格"),
    ("visualize array of arrays of images", "可视化图像数组的数组"),
    ("An implementation of im2col based on some fancy indexing", "基于 fancy indexing 的 im2col 实现"),
    ("An implementation of col2im based on fancy indexing and np.add.at", "基于 fancy indexing 和 np.add.at 的 col2im 实现"),
    ("Performs zero-shot image classification using a CLIP model.", "使用 CLIP 模型执行 zero-shot 图像分类。"),
    ("Compute the mean Intersection over Union (IoU).", "计算平均 Intersection over Union（IoU）。"),
    ("Train the segmentation model using the provided training data.", "使用提供的训练数据训练分割模型。"),
    ("Perform inference on the given test DINO feature vectors.", "在给定测试 DINO 特征向量上执行推理。"),
    ("Upsample the image feature resolution a factor of 2.", "将图像特征分辨率上采样 2 倍。"),
    ("Downsample the image feature resolution a factor of 2.", "将图像特征分辨率下采样 2 倍。"),
    ("RMSNorm layer which is compute-efficient simplified variant of LayerNorm.", "RMSNorm 层，是一种计算高效的 LayerNorm 简化变体。"),
    ("Sinusoidal position embedding for time steps.", "用于时间步的正弦位置嵌入。"),
    ("A conv block with feature modulation.", "带特征调制的卷积 block。"),
    ("A ResNet-like block with context dependent feature modulation.", "带上下文相关特征调制的类 ResNet block。"),
    ("Classifier-free guidance forward pass. model_kwargs should contain `cfg_scale`.", "Classifier-free guidance 前向传播。model_kwargs 应包含 `cfg_scale`。"),
    ("Forward pass through the U-Net.", "通过 U-Net 的前向传播。"),
]


IDENTIFIER_RE = re.compile(r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*")


def looks_code_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("http://", "https://", "www.")):
        return True
    if stripped in {"#", "##", "###"}:
        return True
    words = IDENTIFIER_RE.findall(stripped)
    if len(words) == 1:
        word = words[0].strip("`")
        return word in keyword.kwlist or "_" in word or word.isupper()
    return False


def apply_phrase_replacements(text: str) -> str:
    for old, new in DOC_REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in PHRASES:
        text = text.replace(old, new)
    for old, new in CLEANUP_PHRASES:
        text = text.replace(old, new)
    for old, new in ARTIFACT_REPAIRS:
        text = text.replace(old, new)
    return text


def translate_comment(comment: str) -> str:
    if comment in COMMENT_EXACT:
        return COMMENT_EXACT[comment]
    if comment.strip() in COMMENT_EXACT:
        prefix = comment[: len(comment) - len(comment.lstrip())]
        return prefix + COMMENT_EXACT[comment.strip()]

    marker = ""
    body = comment
    if body.lstrip().startswith("#"):
        prefix_len = len(body) - len(body.lstrip())
        marker = body[:prefix_len] + "#"
        body = body[prefix_len + 1 :]
        if body.startswith(" "):
            marker += " "
            body = body[1:]

    if not body.strip() or set(body.strip()) <= {"#", "-", "*", "="}:
        return comment

    translated = apply_phrase_replacements(body)
    if translated == body and looks_code_like(body):
        return comment
    return marker + translated


def translate_docstring_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.strip():
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        body = line[len(indent) :]
        translated = apply_phrase_replacements(body)
        lines.append(indent + translated)
    return "\n".join(lines)


def docstring_positions(source: str) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return positions
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            positions.add((first.value.lineno, first.value.col_offset))
    return positions


def transform_python_source(source: str) -> str:
    doc_positions = docstring_positions(source)
    result = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            tok = tokenize.TokenInfo(tok.type, translate_comment(tok.string), tok.start, tok.end, tok.line)
        elif tok.type == tokenize.STRING and tok.start in doc_positions:
            try:
                value = ast.literal_eval(tok.string)
            except Exception:
                value = None
            if isinstance(value, str):
                translated = translate_docstring_text(value)
                if tok.string.startswith(('"""', "'''")):
                    quote = tok.string[:3]
                    new_string = quote + translated.replace(quote, "\\" + quote) + quote
                else:
                    new_string = repr(translated)
                tok = tokenize.TokenInfo(tok.type, new_string, tok.start, tok.end, tok.line)
        result.append(tok)
    return tokenize.untokenize(result)


def localize_python_files() -> None:
    for path in sorted(ROOT.glob("assignment[123]/**/*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        localized = transform_python_source(source)
        path.write_text(localized, encoding="utf-8")


def localize_notebook_code_comments() -> None:
    for path in sorted(ROOT.glob("assignment[123]/*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            source = cell.source
            try:
                localized = transform_python_source(source)
            except (tokenize.TokenError, IndentationError):
                continue
            if localized != source:
                cell.source = localized
                changed = True
        if changed:
            nbformat.validate(nb)
            nbformat.write(nb, path)


def main() -> None:
    localize_python_files()
    localize_notebook_code_comments()


if __name__ == "__main__":
    main()
