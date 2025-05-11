import argparse
import torch
import os
from models.encoder import AEncoder, SEncoder
from models.decoder import Decoder
from models.segmentor import Segmentor
from data.dataloader import FundusDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_domain', type=int, default=1, help='测试域的编号')
    parser.add_argument('--output_dir', type=str, default='./output/test_results')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()

class Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 加载模型
        self.load_model()

        # 加载测试数据
        self.test_dataset = FundusDataset(args.data_dir, args.test_domain)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def load_model(self):
        # 创建模型
        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)

        # 从checkpoint获取参数
        model_args = checkpoint['args']

        # 初始化模型
        self.aencoder = AEncoder(
            in_channels=model_args.in_channel,
            out_channels=model_args.anatomy_channel
        ).to(self.device)

        self.sencoder = SEncoder(
            img_size=model_args.img_size,
            style_code_dim=model_args.style_code_dim,
            in_channels=model_args.in_channel
        ).to(self.device)

        self.decoder = Decoder(
            in_channel=model_args.anatomy_channel,
            hidden_channel=256,
            out_channel=model_args.in_channel,
            style_code_dim=model_args.style_code_dim
        ).to(self.device)

        self.segmentor = Segmentor(
            in_channels=model_args.anatomy_channel,
            hidden_size=32,
            num_classes=model_args.num_classes
        ).to(self.device)

        # 加载权重
        self.aencoder.load_state_dict(checkpoint['aencoder'])
        self.sencoder.load_state_dict(checkpoint['sencoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.segmentor.load_state_dict(checkpoint['segmentor'])

        # 设置为评估模式
        self.aencoder.eval()
        self.sencoder.eval()
        self.decoder.eval()
        self.segmentor.eval()

    def save_image(self, tensor, filename):
        # 将tensor转换为PIL图像并保存
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(filename)

    def test(self):
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # 获取数据
                image = data['image'].to(self.device)
                mask = data['mask'].to(self.device)

                # 前向传播
                content_features = self.aencoder(image)
                style_code, mu, logvar = self.sencoder(image)
                seg_pred = self.segmentor(content_features)
                reconstructed = self.decoder(content_features, style_code)

                # 保存结果
                save_dir = os.path.join(self.args.output_dir, f'sample_{i}')
                os.makedirs(save_dir, exist_ok=True)

                # 保存原始图像
                self.save_image(image[0], os.path.join(save_dir, 'original.png'))

                # 保存重建图像
                self.save_image(reconstructed[0], os.path.join(save_dir, 'reconstructed.png'))

                # 保存分割结果
                seg_pred = F.softmax(seg_pred, dim=1)
                seg_pred = seg_pred.argmax(dim=1)[0]
                seg_pred = seg_pred.cpu().numpy()
                plt.imsave(os.path.join(save_dir, 'segmentation.png'), seg_pred, cmap='gray')

                # 保存真实分割标签
                mask = mask[0].cpu().numpy()
                plt.imsave(os.path.join(save_dir, 'ground_truth.png'), mask, cmap='gray')

                print(f'Processed sample {i}')

if __name__ == '__main__':
    args = parse_args()
    tester = Tester(args)
    tester.test()
