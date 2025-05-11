import numpy as np
import torch
import torch.nn.functional as F

def compute_dice(y_pred,y_true):
    """
       y_pred: [B, C, H, W] (after softmax or sigmoid)
       y_true: [B, H, W] (int64 class labels, NOT one-hot)
       """
    num_classes = y_pred.shape[1]
    y_true_onehot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()   # B,H,W,C->B,C,H,W
    smooth = 1e-6

    intersection = (y_pred * y_true_onehot).sum(dim=(2, 3))
    union = (y_pred + y_true_onehot).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)

    return dice.mean()