import torch
import numpy as np


def sim(z_i, z_j):
    """归一化点积 between two vectors.

    输入:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    返回:
    - A scalar 值 该 is 归一化后的 dot product between z_i 并 z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO：你的代码从这里开始。                                                  #
    #                                                                            #
    # 提示： torch.linalg.norm 可能 be helpful.                                  #
    ##############################################################################
    
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """计算 contrastive 损失 L 在 a batch (naive loop version).
    
    输入:
    - out_left: NxD tensor; 输出 的 projection head g(), left branch 在 SimCLR 模型.
    - out_right: NxD tensor; 输出 的 projection head g(), right branch 在 SimCLR 模型.
    Each row is a z-vector 用于 an augmented sample 在 batch. same row 在 out_left 并 out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair 用于 所有 k=0...N-1.
    - tau: scalar 值, temperature parameter 该 determines how fast exponential increases.
    
    返回:
    - A scalar 值; total 损失 across 所有 positive pairs 在 batch. See notebook 用于 definition.
    """
    N = out_left.shape[0]  # total 数量 训练 样本
    
     # Concatenate out_left 并 out_right 到 a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO：你的代码从这里开始。                                                  #
        #                                                                            #
        # 提示： 计算 l(k, k+N) 并 l(k+N, k).                                     #
        ##############################################################################
        
        ##############################################################################
        #                               你的代码结束                             #
        ##############################################################################
    
    # In end, 我们需要 到 divide total 损失 by 2N, 数量 样本 在 batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """归一化点积 between positive pairs.

    输入:
    - out_left: NxD tensor; 输出 的 projection head g(), left branch 在 SimCLR 模型.
    - out_right: NxD tensor; 输出 的 projection head g(), right branch 在 SimCLR 模型.
    Each row is a z-vector 用于 an augmented sample 在 batch.
    same row 在 out_left 并 out_right form a positive pair.
    
    返回:
    - A Nx1 tensor; each row k is 归一化后的 dot product between out_left[k] 并 out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO：你的代码从这里开始。                                                  #
    #                                                                            #
    # 提示： torch.linalg.norm 可能 be helpful.                                  #
    ##############################################################################
    
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """计算 a 2N x 2N 矩阵 的 归一化后的 dot products between 所有 pairs 的 augmented 样本 在 a batch.

    输入:
    - out: 2N x D tensor; each row is z-vector (输出 的 projection head) 的 a single augmented 样本.
    There are a total 的 2N augmented 样本 在 batch.
    
    返回:
    - sim_矩阵: 2N x 2N tensor; each element i, j 在 矩阵 is 归一化后的 dot product between out[i] 并 out[j].
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
    """计算 contrastive 损失 L 在 a batch (向量化版本). 不允许使用循环.
    
    输入 并 输出 are same as 在 simclr_损失_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left 并 out_right 到 a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # 计算 similarity 矩阵 between 所有 pairs 的 augmented 样本 在 batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO：你的代码从这里开始。 Follow hints.                                #
    ##############################################################################
    
    # Step 1: 使用 sim_矩阵 到 计算 denominator 值 用于 所有 augmented 样本.
    # 提示： 计算 e^{sim / tau} 并 存储 到 exponential, which 应该 have 形状 2N x 2N.
    exponential = None
    
    # This binary mask zeros out terms 其中 k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # 提示： 计算 denominator 值 用于 所有 augmented 样本. This 应为 a 2N x 1 vector.
    denom = None

    # Step 2: 计算 similarity between positive pairs.
    # 你可以 do 这个 在 two ways: 
    # Option 1: Extract corresponding indices 来自 sim_矩阵. 
    # Option 2: 使用 sim_positive_pairs().
    
    
    # Step 3: 计算 numerator 值 用于 所有 augmented 样本.
    numerator = None
    
    
    # Step 4: Now 该 you have numerator 并 denominator 用于 所有 augmented 样本, 计算 total 损失.
    loss = None
    
    ##############################################################################
    #                               你的代码结束                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))