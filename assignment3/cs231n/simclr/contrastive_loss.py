import torch
import numpy as np


def sim(z_i, z_j):
    """计算两个向量之间的归一化点积。

    输入:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    返回:
    - 一个 scalar，表示 z_i 和 z_j 之间的归一化点积。
    """
    norm_dot_product = None
    ##############################################################################
    # TODO：你的代码从这里开始。                                                  #
    #                                                                            #
    # 提示：torch.linalg.norm 可能有用。                                        #
    ##############################################################################
    
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """计算一个 batch 上的 contrastive 损失 L（naive loop 版本）。
    
    输入:
    - out_left: NxD tensor；SimCLR 模型左分支中 projection head g() 的输出。
    - out_right: NxD tensor；SimCLR 模型右分支中 projection head g() 的输出。
    每一行都是 batch 中一个 augmented sample 的 z-vector。out_left 和 out_right
    中相同行构成一个 positive pair。换言之，对所有 k=0...N-1，
    (out_left[k], out_right[k]) 构成一个 positive pair。
    - tau: scalar，temperature parameter，决定 exponential 增长速度。
    
    返回:
    - 一个 scalar；batch 中所有 positive pairs 的总损失。定义见 notebook。
    """
    N = out_left.shape[0]  # total 数量 训练 样本
    
     # 将 out_left 和 out_right 拼接成一个 2*N x D tensor。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO：你的代码从这里开始。                                                  #
        #                                                                            #
        # 提示：计算 l(k, k+N) 和 l(k+N, k)。                                    #
        ##############################################################################
        
        ##############################################################################
        #                               你的代码结束                             #
        ##############################################################################
    
    # 最后，需要将 total_loss 除以 2N，即 batch 中样本总数。
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """计算 positive pairs 之间的归一化点积。

    输入:
    - out_left: NxD tensor；SimCLR 模型左分支中 projection head g() 的输出。
    - out_right: NxD tensor；SimCLR 模型右分支中 projection head g() 的输出。
    每一行都是 batch 中一个 augmented sample 的 z-vector。
    out_left 和 out_right 中相同行构成一个 positive pair。
    
    返回:
    - 一个 Nx1 tensor；第 k 行是 out_left[k] 和 out_right[k] 之间的归一化点积。
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO：你的代码从这里开始。                                                  #
    #                                                                            #
    # 提示：torch.linalg.norm 可能有用。                                        #
    ##############################################################################
    
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """计算 2N x 2N 矩阵，包含 batch 中所有 augmented sample pair 的归一化点积。

    输入:
    - out: 2N x D tensor；每一行是单个 augmented sample 的 z-vector
      （projection head 的输出）。batch 中共有 2N 个 augmented samples。
    
    返回:
    - sim_matrix: 2N x 2N tensor；矩阵中元素 (i, j) 是 out[i] 和 out[j]
      之间的归一化点积。
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO：你的代码从这里开始。                                                  #
    ##############################################################################
    

    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """计算一个 batch 上的 contrastive 损失 L（向量化版本）。不允许使用循环。
    
    输入和输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D tensor。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # 计算 batch 中所有 augmented sample pair 之间的 similarity 矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO：你的代码从这里开始。按照提示完成。                                  #
    ##############################################################################
    
    # Step 1: 使用 sim_matrix 计算所有 augmented samples 的 denominator。
    # 提示：计算 e^{sim / tau} 并存入 exponential，它的形状应为 2N x 2N。
    exponential = None
    
    # 这个二值 mask 会将 k=i 的项置零。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # 应用二值 mask。
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # 提示：计算所有 augmented samples 的 denominator。它应为 2N x 1 vector。
    denom = None

    # Step 2: 计算 positive pairs 之间的 similarity。
    # 你可以用两种方式完成：
    # Option 1: 从 sim_matrix 中提取对应索引。
    # Option 2: 使用 sim_positive_pairs()。
    
    
    # Step 3: 计算所有 augmented samples 的 numerator。
    numerator = None
    
    
    # Step 4: 现在已经有所有 augmented samples 的 numerator 和 denominator，计算总损失。
    loss = None
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
