import torch.nn as nn
import torch
from .encoder import SEncoder,AEncoder
from .block import MLP
import torch.nn.functional as F

class AdaIN(nn.Module):
    '''
        input: gamma(SRM生成),beta,Fic(上一层)
        gama shape：[B,num_feature]/[B,C]
        beta shape:[B,num_feature]/[B,C]
        Fic shape:[B,C,H,W]
    '''
    def __init__(self,eps=1e-6):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, x, gamma, beta):
        # x: [B, C, H, W]
        assert gamma.shape[1] == x.shape[1] and beta.shape[1]==x.shape[1], "SRM out_channel must be the same with AdaIN in_channel"
        B, C = x.size(0), x.size(1)

        # 每个样本、每个通道计算 mean 和 std（instance norm）
        mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        std = x.view(B, C, -1).std(dim=2, unbiased=False).view(B, C, 1, 1)

        # AdaIN: 归一化 + 放缩 + 平移
        out = (x - mean) / (std + self.eps)
        out = gamma.view(B, C, 1, 1) * out + beta.view(B, C, 1, 1)

        return out

class SRM(nn.Module):
    def __init__(self, style_code_dim, num_features, hidden_dim=256):
        """
        Args:
            style_code_dim: 输入的风格码维度（如 8 或 16）
            num_features: 输出 AdaIN 所需通道数
            hidden_dim: MLP 中间层维度
        """
        super(SRM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2)  # 输出 gamma 和 beta 拼接
        )

    def forward(self, style_code):
        out = self.fc(style_code)
        gamma, beta = out.chunk(2, dim=1)  # 拆分为两个 [B, num_features]
        return gamma, beta

class Decoder(nn.Module):
    '''
        根据style code和seg重构图片
    '''
    def __init__(self,in_channel,hidden_channel=256,out_channel=3,style_code_dim=16):
        super(Decoder, self).__init__()

        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

        self.conv1=nn.Conv2d(self.in_channel,self.hidden_channel,kernel_size=3,stride=1,padding=1,bias=True) # 不改变大小
        self.conv2 = nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv3 = nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv4 = nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3, stride=1, padding=1,bias=True)

        self.SRM1 = SRM(style_code_dim=style_code_dim, num_features=hidden_channel)
        self.SRM2 = SRM(style_code_dim=style_code_dim, num_features=hidden_channel)
        self.SRM3 = SRM(style_code_dim=style_code_dim, num_features=hidden_channel)

        self.adain=AdaIN()

    def forward(self,x_seg,style_code):
        '''
            todo:i表示某个域第i组？一组三张图片？是一张图片一个gama和beta？  应该不是，因为论文中有张图d=i时，里面有很多点，i应该是一个batch
            确实是一张图片产生三个相同的beta和gamma
        '''
        x = self.conv1(x_seg)
        gama1,beta1=self.SRM1(style_code)
        x=self.adain(x,gama1,beta1)

        x=self.conv2(x)
        gama2,beta2=self.SRM2(style_code)
        x=self.adain(x,gama2,beta2)

        x = self.conv3(x)
        gama3, beta3 = self.SRM3(style_code)
        x = self.adain(x, gama3, beta3)

        x = self.conv4(x)
        x=torch.tanh(x)  # 归一化到[-1,1]

        return x

    def reconstructed_loss(self,x,reconstructed_x):
        return nn.L1Loss()(x,reconstructed_x).item()

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    encoder = AEncoder(in_channels=3, hidden_channels=32, out_channels=16)
    out = encoder(x) # [2,16,256,256]
    print(out.shape)

    style_code_dim=16
    sencoder = SEncoder(img_size=256, style_code_dim=style_code_dim, in_channels=3, out_channels=32, hidden_channels=16)
    out1 = sencoder(x)  # 必须batch>2  [2,16]
    decoder = Decoder(in_channel=3,hidden_channel=256,out_channel=3,style_code_dim=style_code_dim)
    out=decoder(x,out1[0])
    print(out.shape)  # [2,3,256,256]
    print(decoder.reconstructed_loss(x,out))