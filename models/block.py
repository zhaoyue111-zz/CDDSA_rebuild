from torch import nn
import torch
import torch.nn.functional as F

# AEncoder基于Unet，设置最后几层的输出维度位T，使用tanh作为激活函数
class Unet(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        '''
        input H,W,T
        output H,W,T
        '''
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # DownSample
        self.down1=self.conv_relu(self.in_channels,self.hidden_channels)
        self.down2=self.conv_relu(self.hidden_channels,self.hidden_channels*2)
        self.down3=self.conv_relu(self.hidden_channels*2,self.hidden_channels*4)
        self.down4=self.conv_relu(self.hidden_channels*4,self.hidden_channels*8)
        self.down5=self.conv_relu(self.hidden_channels*8,self.hidden_channels*16)

        self.maxpool=nn.MaxPool2d(kernel_size=2)  # 2倍下采样

        # UpSample
        self.up1 = nn.ConvTranspose2d(hidden_channels * 16, hidden_channels * 8, kernel_size=2, stride=2)
        self.conv1 = self.conv_relu(hidden_channels * 16, hidden_channels * 8)

        self.up2 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2)
        self.conv2 = self.conv_relu(hidden_channels * 8, hidden_channels * 4)

        self.up3 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2)
        self.conv3 = self.conv_relu(hidden_channels * 4, hidden_channels * 2)

        self.up4 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv4 = self.conv_relu(hidden_channels * 2, hidden_channels)

        self.output_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def conv_relu(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def crop_tensor(self, source, target):
        """中心裁剪 source，使其与 target 尺寸一致"""
        _, _, h, w = target.size()
        _, _, hs, ws = source.size()
        dh = (hs - h) // 2
        dw = (ws - w) // 2
        return source[:, :, dh:dh + h, dw:dw + w]


    def forward(self,x):
        x1 = self.down1(x)  # HxW
        x2 = self.down2(self.maxpool(x1))  # H/2 x W/2
        x3 = self.down3(self.maxpool(x2))  # H/4 x W/4
        x4 = self.down4(self.maxpool(x3))  # H/8 x W/8
        x5 = self.down5(self.maxpool(x4))  # H/16 x W/16

        up = self.up1(x5)
        res = torch.cat([up, self.crop_tensor(x4, up)], dim=1)
        res = self.conv1(res)

        up = self.up2(res)
        res = torch.cat([up, self.crop_tensor(x3, up)], dim=1)
        res = self.conv2(res)

        up = self.up3(res)
        res = torch.cat([up, self.crop_tensor(x2, up)], dim=1)
        res = self.conv3(res)

        up = self.up4(res)
        res = torch.cat([up, self.crop_tensor(x1, up)], dim=1)
        res = self.conv4(res)

        return self.output_layer(res)


def conv_bn_lrelu(in_channels,out_channels,kernel_size=3,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2,inplace=True)
    )

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = F.relu(out, inplace=True)
        out = self.fc3(out)
        out = F.relu(out, inplace=True)

        return out

if __name__ == '__main__':
    model = Unet(in_channels=3, hidden_channels=32, out_channels=16)  # T=16
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)  # Should be [1, 16, 256, 256]