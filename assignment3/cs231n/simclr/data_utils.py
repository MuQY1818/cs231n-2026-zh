from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    此函数返回作用于单张训练图像的一组数据增强组合。
    补全下面的代码。提示：查看 torchvision.transforms 中可用的函数。
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # 应用 color jitter 的变换：brightness=0.4、contrast=0.4、saturation=0.4、hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # TODO：你的代码从这里开始。                                                  #
        #                                                                            #
        # 提示：查看 torchvision.transforms 中定义的 transformation 函数。      #
        # 第一个操作已经作为示例为你填好。                                      #
        ##############################################################################
        # Step 1: 随机 resize 并 crop 到 32x32。
        transforms.RandomResizedCrop(32),
        # Step 2: 以 0.5 的概率水平翻转 image。

        # Step 3: 以 0.8 的概率应用 color jitter（可以使用上面定义的 "color_jitter"）。

        # Step 4: 以 0.2 的概率将 image 转换为灰度。

        ##############################################################################
        #                               你的代码结束                             #
        ##############################################################################
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO：你的代码从这里开始。                                                  #
            #                                                                            #
            # 对 image 应用 self.transform，生成论文中的 x_i 和 x_j。 #
            ##############################################################################
            pass
            ##############################################################################
            #                               你的代码结束                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
