import torch
from .block import Unet,conv_bn_lrelu
import torch.nn as nn

# 基于Unet
class AEncoder(nn.Module):
    def __init__(self,in_channels=3,hidden_channels=16,out_channels=8):
        super(AEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.unet=Unet(self.in_channels,self.hidden_channels,self.out_channels)

    def forward(self, x):
        x=self.unet(x)
        x=torch.tanh(x)
        return x

# VAE
class SEncoder(nn.Module):
    '''
        根据给定的源域图片，预测latent code(Z)(假定其符合(1,0)的高斯分布)分布的均值和方差
        一个图片的talent code是从符合该域的均值和方差的高斯分布中采样得到

        input shape:(B,C,H,W)
    '''
    def __init__(self,img_size,style_code_dim=16,in_channels=3,out_channels=32,hidden_channels=16):
        super(SEncoder, self).__init__()

        self.img_size=img_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.style_code_dim = style_code_dim

        self.block1 = conv_bn_lrelu(self.in_channels, self.hidden_channels, 3, 2, 1)
        self.block2 = conv_bn_lrelu(self.hidden_channels, self.hidden_channels*2, 3, 2, 1)
        self.block3 = conv_bn_lrelu(self.hidden_channels*2, self.hidden_channels*4, 3, 2, 1)
        self.block4 = conv_bn_lrelu(self.hidden_channels*4, self.hidden_channels*8, 3, 2, 1)

        self.output_layer=nn.Sequential(
            nn.Linear(self.hidden_channels*8*self.img_size//16*self.img_size//16, self.out_channels),
            nn.BatchNorm1d(self.out_channels),  # 要求训练模式下batch_size>=2  因为batch_size=1时，计算的方差=0，无法归一化
            nn.ReLU(inplace=True)
        )

        self.mu=nn.Linear(self.out_channels,self.style_code_dim)
        self.logvar=nn.Linear(self.out_channels,self.style_code_dim)  # 使用log：数值稳定，避免负方差

    # 重参数化
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        x = self.block1(x)  # 1,16,128,128
        x = self.block2(x) # 1,32,64,64
        x = self.block3(x) # 1,64,32,32
        x = self.block4(x)  # 1,128,16,16

        x=x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])  # 1,128*16*16=32768
        out=self.output_layer(x)

        mu=self.mu(out)
        logvar=self.logvar(out)

        z=self.reparameterize(mu,logvar)
        return z,mu,logvar

    def KL_loss(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl / mu.size(0)


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    encoder = AEncoder(in_channels=3, hidden_channels=32, out_channels=16)
    out = encoder(x)
    print(out.shape)

    sencoder=SEncoder(img_size=256,style_code_dim=16, in_channels=3,out_channels=32, hidden_channels=16)
    out1=sencoder(x)  # 必须batch>2
    print(out1[0].shape)  # z
    print(out1[1].shape)  # mu
    print(out1[2].shape)  # var
    print(sencoder.KL_loss(out1[1],out1[2]))