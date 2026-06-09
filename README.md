# CS231n 2026 作业中文本地化

这是 Stanford CS231n Spring 2026 三个作业 starter code 的非官方中文本地化版本。内容来自官方作业包，本仓库只做中文说明、notebook 文本、代码注释和 docstring 本地化；作业 TODO 和参考答案没有补全。

官方页面：https://cs231n.stanford.edu/assignments.html

## 内容

- `assignment1/`：kNN、SVM、Softmax、两层网络、全连接网络、特征提取。
- `assignment2/`：BatchNorm、Dropout、卷积网络、PyTorch、RNN captioning。
- `assignment3/`：Transformer captioning、CLIP/DINO、SimCLR、DDPM。
- `tools/`：本地化脚本，供后续同步或修正时使用。

说明：变量名、库名、函数名、张量形状和常见技术词会保留英文，以免影响理解代码和搜索文档。

## 环境配置

不要在 conda `base` 环境里安装依赖。建议单独建一个环境：

```bash
conda create -n cs231n-zh python=3.10 -y
conda activate cs231n-zh
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib imageio future six notebook ipykernel ipywidgets h5py pillow tqdm
python -m ipykernel install --user --name cs231n-zh --display-name "Python (cs231n-zh)"
```

Assignment 2/3 中的 PyTorch notebook 还需要安装 PyTorch。按你的 CUDA/CPU 环境选择官方安装命令；如果只做 CPU 版，可以先用：

```bash
python -m pip install torch torchvision
```

`assignment3/requirements.txt` 是官方原包附带的旧版本依赖清单，其中很多 pin 比较老。新环境建议先用上面的最小依赖启动，遇到缺包再补，不建议直接把旧清单装进已有环境。

## 下载数据

仓库不包含 CIFAR-10、COCO、ImageNet 校验数据、模型权重或训练输出。需要时进入对应目录运行官方下载脚本：

```bash
cd assignment1/cs231n/datasets
bash get_datasets.sh
```

```bash
cd assignment2/cs231n/datasets
bash get_datasets.sh
bash get_coco_dataset.sh
```

```bash
cd assignment3/cs231n/datasets
bash get_datasets.sh
bash get_coco_dataset.sh
```

## 启动 Jupyter

在仓库根目录启动：

```bash
conda activate cs231n-zh
jupyter notebook --notebook-dir .
```

浏览器打开后，先选择 `assignment1/knn.ipynb` 开始。notebook 右上角 kernel 请选择 `Python (cs231n-zh)`。

推荐 Assignment 1 顺序：

1. `assignment1/knn.ipynb`
2. `assignment1/softmax.ipynb`
3. `assignment1/two_layer_net.ipynb`
4. `assignment1/FullyConnectedNets.ipynb`
5. `assignment1/features.ipynb`

## 注意事项

- 本仓库不是官方课程材料，也不是答案仓库。
- 请遵守 CS231n 课程的 Honor Code 和作业协作政策。
- 不要提交数据集、模型权重、训练输出、`.ipynb_checkpoints` 或 `__pycache__`。
- 如果 notebook 中仍有少量英文，一般是变量名、技术术语、命令、URL 或张量形状说明。
