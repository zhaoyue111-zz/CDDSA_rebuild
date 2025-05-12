import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .decoder import Decoder

def DomainStyleContrastiveLoss(style_codes, domain_labels,temperature=0.1):
    '''
        style_codes: [B, style_dim]
        domain_labels: [B]
    '''
    # compute number of sample in each domain
    unique_domains = torch.unique(domain_labels)
    # style_codes
    features = F.normalize(style_codes, dim=1)

    total_loss = 0
    total_valid_pairs=0
    for domain_idx in unique_domains: # 对每个域
        domain_mask = (domain_labels == domain_idx)  # 必须从0开始
        domain_samples = features[domain_mask]
        num_samples=len(domain_samples)
        if num_samples < 2:
            continue

        # create positive sample pair
        # perm_idx = torch.randperm(num_samples) 有可能产生和Q一样的Q_tilde
        Q = domain_samples  # 本域的风格编码集
        shift = torch.randint(1, num_samples, (1,)).item()  # 随机偏移量 (1 ≤ shift < num_samples)  避免自己和自己计算相似性
        perm_idx = (torch.arange(num_samples) + shift) % num_samples
        Q_tilde = domain_samples[perm_idx]  # 打乱顺序


        for i in range(num_samples):  # 本域中所有图片
            # 两个正样本(本域中随机两个样本)
            anchor = Q[i:i+1]  # [1, style_dim]
            positive = Q_tilde[i:i+1]  # [1, style_dim]

            # 对该域来说的所有负样本
            negative_mask = (domain_labels != domain_idx)
            negatives = features[negative_mask]  # [b(D-1), style_dim]

            pos_sim = torch.sum(anchor * positive, dim=1) / temperature
            neg_sim = torch.matmul(anchor, negatives.T).squeeze(0) / temperature

            # InfoNCE loss
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sim))
            loss = -torch.log(numerator / denominator)
            total_loss += loss

            total_valid_pairs+=1

    if total_valid_pairs == 0:
        return torch.tensor(0.0, device=style_codes.device)

    return total_loss / total_valid_pairs


def StyleAugmentationLoss(x_seg, style_codes):
    '''
        x_seg:[B,C,H,W]
        style_codes:[B,style_code_dim]
        temperature:控制权重分布的尖锐程度
    '''
    B, C, H, W = x_seg.shape
    style_dim = style_codes.size(1)

    # 生成均匀随机权重alpha [B, B]（每行独立采样）
    alpha = torch.rand(B, B, device=style_codes.device)  # [0, 1)均匀分布
    alpha = alpha / alpha.sum(dim=1, keepdim=True)

    new_style_codes = torch.matmul(alpha, style_codes) # [B,style_code_dim]

    decoder = Decoder(
        in_channel=C,
        hidden_channel=256,
        out_channel=C,
        style_code_dim=style_dim
    )
    out = decoder(x_seg, new_style_codes)  # [B,C,H,W]

    return nn.L1Loss()(x_seg, out)


if __name__ == '__main__':
    # num_domains = 3  # 3个域
    # samples_per_domain = 4  # 每个域4个样本
    # style_dim = 16  # 风格编码维度
    # batch_size = num_domains * samples_per_domain
    #
    # 随机生成风格编码
    # style_codes_domain1 = torch.randn(2, style_dim)
    # style_codes_domain2 = torch.randn(3, style_dim)
    # style_codes_domain3= torch.randn(4, style_dim)
    # style_codes=torch.cat((style_codes_domain1,style_codes_domain2,style_codes_domain3),dim=0)  # [1,16]

    # 生成域标签 [0,0,0,0,1,1,1,1,2,2,2,2]
    # domain_labels = torch.tensor([0,0,1,1,1,2,2,2,2])
    # loss = DomainStyleContrastiveLoss(style_codes, domain_labels,temperature=0.1)
    #
    # print(f"生成的数据形状:")
    # print(f"风格编码: {style_codes.shape}")
    # print(f"域标签: {domain_labels.shape}")
    # print(f"计算的损失值: {loss.item():.4f}")

    x_seg = torch.randn(8, 3, 256, 256)  # 8张图片的特征
    style_codes = torch.randn(8, 16)  # 8个风格编码
    loss = StyleAugmentationLoss(x_seg, style_codes)
    print(loss.item())