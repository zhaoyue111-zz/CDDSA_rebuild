import torch
import torch.nn as nn
from .matrix import compute_dice
from .encoder import AEncoder
from .block import conv_bn_lrelu

class Segmentor(nn.Module):
    '''
        input shape: (B,C,W,H)
        output shape: (B,num_class,W,H)
    '''
    def __init__(self,in_channels,hidden_size,num_classes=3):
        super(Segmentor,self).__init__()

        # 论文中
        self.conv1 = conv_bn_lrelu(in_channels,hidden_size,kernel_size=3,stride=1,padding=1)
        self.conv2 = conv_bn_lrelu(hidden_size,hidden_size,kernel_size=1,stride=1,padding=0)
        self.out=nn.Conv2d(hidden_size,num_classes,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.out(x2)

    # 计算Lseg
    def Seg_loss(self,y_pred,y_true):
        """
            y_pred: [B, C, H, W] (logits)
            y_true: [B, H, W] (class labels)
            """
        criterion = nn.CrossEntropyLoss()
        cross_entropy = criterion(y_pred, y_true.long())  # 只能接受long类型的标签
        dice = compute_dice(torch.softmax(y_pred, dim=1), y_true)  # 0-1 scalar
        return 0.5 * (cross_entropy + dice)  # dice 越大越好，loss 应该减它


if __name__ == '__main__':
    model = AEncoder(in_channels=3, hidden_channels=32, out_channels=16)  # T=16
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)  # Should be [1, 16, 256, 256]
    seg=Segmentor(out.shape[1],hidden_size=32,num_classes=3)
    out=seg(out)
    print(out.shape)  # should be [1, 3, 256, 256]

    y_true = torch.randint(low=0, high=3, size=(1, 256, 256))  # 不能为负数，只能生成0 1 2
    dice=seg.Seg_loss(out,y_true)
    print(dice)