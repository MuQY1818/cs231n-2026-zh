from tensorflow.python.framework.ops import device_v2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    计算 text feature vector 和 image feature vector 之间的 pairwise cosine similarity。

    参数:
        text_features (torch.Tensor): 形状为 (N, D) 的 tensor.
        image_features (torch.Tensor): 形状为 (M, D) 的 tensor.

    返回:
        torch.Tensor: 形状为 (N, M) 的 similarity 矩阵，其中每个元素 (i, j)
        是 text_features[i] 和 image_features[j] 之间的 cosine similarity。
    """
    similarity = None
    ############################################################################
    # TODO: 计算 cosine similarity。不要使用 for loop。                  #
    ############################################################################

    ############################################################################
    #                             你的代码结束                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """使用 CLIP 模型执行 zero-shot 图像分类。

    参数:
        clip_model (torch.nn.Module): 用于编码 images 和 text 的 pretrained
            CLIP 模型.
        clip_preprocess (Callable): 编码前应用到每张 image 的 preprocessing 函数.
        images (List[np.ndarray]): 输入 images 列表，每张为 (H x W x C) 的
            uint8 NumPy 数组.
        class_texts (List[str]): zero-shot 分类使用的类别标签字符串列表.
        device (torch.device): 执行计算的 device。将 text_tokens 传入
            clip_model 前，应先移动到这个 device.

    返回:
        List[str]: 每张 image 的预测类别标签，从给定 class_texts 中选择。
    """
    
    pred_classes = []

    ############################################################################
    # TODO: 为 images 找到对应的类别标签。                               #
    ############################################################################

    ############################################################################
    #                             你的代码结束                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    使用 CLIP 的简单 image retrieval system。
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        参数:
          clip_model (torch.nn.Module): pretrained CLIP 模型.
          clip_preprocess (Callable): 用于预处理 images 的函数.
          images (List[np.ndarray]): image 列表，每张为 NumPy 数组 (H x W x C).
          device (torch.device): 模型执行所用 device.
        """
        ############################################################################
        # TODO: 存储 retrieve 方法需要的所有对象变量。                           #
        # 注意：你应该在这里一次性处理所有 images，避免对每个 text query         #
        # 重复计算。为了获得更优计算效率，你最终可以不使用上面的 similarity      #
        # 函数。                                                                 #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        检索与输入 text 最相似的 top-k images 的索引。
        torch.Tensor.topk 方法可能有用。

        参数:
            query (str): text query.
            k (int): 返回 top k images.

        返回:
            List[int]: top-k 最相似 images 的索引.
        """
        top_indices = []
        ############################################################################
        # TODO: 检索 top-k images 的索引。                                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    在 RGB image 上生成彩色 segmentation overlay。

    Parameters:
        segmentation_mask (np.ndarray): 形状为 (H, W) 的 2D 数组，存放类别索引.
        image (np.ndarray): 形状为 (H, W, 3) 的 3D 数组，RGB image.
        alpha (float): overlay 透明度系数，0 表示只显示 image，1 表示只显示 mask.

    返回:
        np.ndarray: 带 segmentation overlay 的 image，形状为 (H, W, 3)，dtype 为 uint8.
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # 使用固定 colormap 为每个类别生成确定性颜色。
    def generate_colormap(n):
        np.random.seed(42)  # 确保结果确定
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # 为 segmentation mask 创建彩色图像。
    seg_color = colormap[segmentation_mask]  # 形状: (H, W, 3)

    # 与原始 image 混合。
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """计算平均 Intersection over Union（IoU）。"""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        初始化 DINOSegmentation 模型。

        这里定义一个简单神经网络，用于把 DINO feature vector 分类为
        segmentation 类别，并完成模型、optimizer 和损失函数的初始化。

        参数:
            device (torch.device): 运行模型的 device（CPU 或 CUDA）。
            num_classes (int): segmentation 类别数量。
            inp_dim (int, optional): 输入 DINO feature 的维度。
        """

        ############################################################################
        # TODO: 定义一个非常轻量的 PyTorch 模型、optimizer 和损失函数，用于      #
        # 训练模型把每个 DINO feature vector 分类到一个 segmentation 类别。       #
        # 它可以是一个线性层，也可以是两层 neural network。                      #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """使用提供的训练数据训练分割模型。

        参数:
            X_train (torch.Tensor): 输入 feature vectors，形状为 (N, D).
            Y_train (torch.Tensor): Ground truth labels，形状为 (N,).
            num_iters (int, optional): optimization step 数量。
        """
        ############################################################################
        # TODO：训练你的模型，共进行 `num_iters` step。                          #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """在给定测试 DINO 特征向量上执行推理。

        参数:
            X_test (torch.Tensor): 输入 feature vectors，形状为 (N, D).

        返回:
            形状为 (N,) 的 torch.Tensor：预测类别索引。
        """
        pred_classes = None
        ############################################################################
        # TODO：使用你的模型预测类别。                                           #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return pred_classes
