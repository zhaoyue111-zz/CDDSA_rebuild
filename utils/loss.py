import torch.nn as nn
import torch
from .matrix import compute_dice

def Reconstructed_loss(x, reconstructed_x):
    return nn.L1Loss()(x, reconstructed_x)


def KL_loss(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl / mu.size(0)


def Seg_loss(y_pred,y_true):
    """
        y_pred: [B, C, H, W] (logits)
        y_true: [B, H, W] (class labels)
        """
    criterion = nn.CrossEntropyLoss()
    cross_entropy = criterion(y_pred, y_true.long())  # 只能接受long类型的标签
    dice = compute_dice(torch.softmax(y_pred, dim=1), y_true)  # 0-1 scalar
    dice_loss=1-dice
    return 0.5 * (cross_entropy + dice_loss)  # dice 越大越好，loss 应该减它


def StyleAugmentationLoss(x_seg, style_codes,Decoder,Encoder):
        '''
            x_seg:[B,C,H,W]
            style_codes:[B,style_code_dim]
        '''
        B, C, H, W = x_seg.shape

        # 生成均匀随机权重alpha [B, B]（每行独立采样）
        alpha = torch.rand(B, B, device=style_codes.device)  # [0, 1)均匀分布
        alpha = alpha / alpha.sum(dim=1, keepdim=True)

        new_style_codes = torch.matmul(alpha, style_codes)  # [B,style_code_dim]

        out = Decoder(x_seg, new_style_codes)  # [B,C,H,W]
        out_rec=Encoder(out)  # [B,C,H,W]

        return nn.L1Loss()(x_seg, out_rec)