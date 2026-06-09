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
    计算 pairwise cosine similarity between text 并 image 特征 vectors.

    参数:
        text_特征 (torch.Tensor): A tensor 的 形状 (N, D).
        image_特征 (torch.Tensor): A tensor 的 形状 (M, D).

    返回:
        torch.Tensor: A similarity 矩阵 的 形状 (N, M), 其中 each entry (i, j)
        is cosine similarity between text_特征[i] 并 image_特征[j].
    """
    similarity = None
    ############################################################################
    # TODO: 计算 cosine similarity. Do NOT 使用 用于 loops.               #
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
        clip_模型 (torch.nn.Module): pre-训练ed CLIP 模型 用于 encoding
            images 并 text.
        clip_preprocess (Callable): A preprocessing 函数 到 apply 到 each
            image before encoding.
        images (List[np.nd数组]): A list 的 输入 images as NumPy 数组
            (H x W x C) uint8.
        类别_texts (List[str]): A list 的 类别 标签 strings 用于 zero-shot
            分类.
        device (torch.device): device on which computation 应为
            performed. Pass text_tokens 到 这个 device before passing it to
            clip_模型.

    返回:
        List[str]: Predicted 类别 标签 用于 each image, selected 来自 the
            given 类别_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find 类别 标签 用于 images.                                  #
    ############################################################################

    ############################################################################
    #                             你的代码结束                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system 使用 CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        参数:
          clip_模型 (torch.nn.Module): pre-训练ed CLIP 模型.
          clip_preprocess (Callable): Function 到 preprocess images.
          images (List[np.nd数组]): List 的 images as NumPy 数组 (H x W x C).
          device (torch.device): device 用于 模型 execution.
        """
        ############################################################################
        # TODO: 存储 所有 necessary object 变量 到 使用 在 retrieve method.    #
        # Note 该 你应该 process 所有 images at once here 并 avoid repeated  #
        # computation 用于 each text query. 你可以 end up NOT 使用 above      #
        # similarity 函数 用于 most 计算-optimal 实现.#
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves indices 的 top-k images most similar 到 输入 text.
        你可以 找到 torch.Tensor.topk method 使用ful.

        参数:
            query (str): text query.
            k (int): Return top k images.

        返回:
            List[int]: Indices 的 top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve indices 的 top-k images.                              #
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
    Generate a colored segmentation 在lay on top 的 an RGB image.

    Parameters:
        segmentation_mask (np.nd数组): 2D 数组 的 形状 (H, W), 使用 类别 indices.
        image (np.nd数组): 3D 数组 的 形状 (H, W, 3), RGB image.
        alpha (float): Transparency 因子 用于 在lay (0 = only image, 1 = only mask).

    返回:
        np.nd数组: Image 使用 segmentation 在lay (形状: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors 用于 each 类别 使用 a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image 用于 segmentation mask
    seg_color = colormap[segmentation_mask]  # 形状: (H, W, 3)

    # Blend 使用 original image
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """计算平均 Intersection 在 Union（IoU）。"""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        初始化 DINOSegmentation 模型.

        This defines a simple neural network designed 到  类别ify DINO 特征
        vectors 到 segmentation 类别. It 包含 模型 initialization,
        optimizer, 并 损失 函数 setup.

        参数:
            device (torch.device): Device 到 run 模型 on (CPU or CUDA).
            num_类别 (int): 数量 segmentation 类别.
            inp_dim (int, optional): Dimensionality 的 输入 DINO 特征.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch 模型, optimizer, 并 损失       #
        # 函数 到 训练 类别ify each DINO 特征 vector 到 a seg. 类别.   #
        # It 可以 be a linear 层 or two 层 neural network.                    #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """使用提供的训练数据训练分割模型。

        参数:
            X_训练 (torch.Tensor): 输入 特征 vectors 的 形状 (N, D).
            Y_训练 (torch.Tensor): Ground truth 标签 的 形状 (N,).
            num_iters (int, optional): 数量 optimization steps.
        """
        ############################################################################
        # TODO：训练 your 模型 用于 `num_iters` steps.                            #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """在给定测试 DINO 特征向量上执行推理。

        参数:
            X_测试 (torch.Tensor): 输入 特征 vectors 的 形状 (N, D).

        返回:
            torch.Tensor 的 形状 (N,): Predicted 类别 indices.
        """
        pred_classes = None
        ############################################################################
        # TODO：训练 your 模型 用于 `num_iters` steps.                            #
        ############################################################################

        ############################################################################
        #                             你的代码结束                             #
        ############################################################################
        return pred_classes