from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This 函数 returns a composition 的 数据增强s 到 a single 训练 image.
    Complete following lines. 提示： look at available 函数 在 torchvision.transforms
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # 应用 color jitter 的变换：brightness=0.4、contrast=0.4、saturation=0.4、hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # TODO：你的代码从这里开始。                                                  #
        #                                                                            #
        # 提示： Check out transformation 函数 defined 在 torchvision.transforms #
        # first operation is filled out 用于 you as an 样本.
        ##############################################################################
        # Step 1: Randomly resize 并 crop 到 32x32.
        transforms.RandomResizedCrop(32),
        # Step 2: Horizont所有y flip image 使用 probability 0.5

        # Step 3: With a probability 的 0.8, apply color jitter (你可以 使用 "color_jitter" defined above.

        # Step 4: With a probability 的 0.2, convert image 到 灰度

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
            # Apply self.transform 到 image 到 produce x_i 并 x_j 在 paper #
            ##############################################################################
            pass
            ##############################################################################
            #                               你的代码结束                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target