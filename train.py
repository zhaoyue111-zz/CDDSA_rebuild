import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from models.encoder import AEncoder, SEncoder
from models.decoder import Decoder
from models.segmentor import Segmentor
from data.dataloader import MultiDomainIterator
from torch.utils.tensorboard import SummaryWriter
from models.loss import DomainStyleContrastiveLoss, StyleAugmentationLoss
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--data_dir', type=str, default='/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_domain', nargs='+', type=int, default=[0, 2, 3],
                        help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--test_domain', nargs='+', type=int, default=[1],
                        help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--sample_list', nargs='+', type=int, default=[4, 4, 10])
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--style_code_dim', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=8, help="the out channel of anatomy encoder")
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=0.001)
    parser.add_argument('--lambda_dsct', type=float, default=0.01)
    parser.add_argument('--lambda_aug', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epoches', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_patience', type=float, default=8)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=25, help='the interval to save checkpoint')
    parser.add_argument('--log_dir', type=str, default='./output/logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/checkpoints')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的checkpoint路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='恢复训练的起始epoch')

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        self.build_model()
        self.build_optimizer()

        # 加载checkpoint（如果存在）
        self.start_epoch = args.start_epoch
        if args.resume:
            self.load_checkpoint(args.resume)

        # 数据加载器
        self.dataloaders = MultiDomainIterator(
            args.data_dir,
            args.train_domain,
            args.sample_list,
        )

        # tensorboard
        self.writer = SummaryWriter(args.log_dir)

        # 初始化最佳损失和没有改善的轮数
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def build_model(self):
        # 创建模型
        self.aencoder = AEncoder(
            in_channels=self.args.in_channel,
            hidden_channels=16,  # 论文中
            out_channels=self.args.anatomy_channel  # 论文中
        ).to(self.device)

        self.sencoder = SEncoder(
            img_size=self.args.img_size,
            style_code_dim=self.args.style_code_dim,
            in_channels=self.args.in_channel
            # out_channels=32,
            # hidden_channels=16
        ).to(self.device)

        self.decoder = Decoder(
            in_channel=self.args.anatomy_channel,
            hidden_channel=256,
            out_channel=self.args.in_channel,
            style_code_dim=self.args.style_code_dim
        ).to(self.device)

        self.segmentor = Segmentor(
            in_channels=self.args.anatomy_channel,
            hidden_size=32,
            num_classes=self.args.num_classes
        ).to(self.device)

    def build_optimizer(self):
        # 负责内容
        self.opt_ae = optim.Adam(
            list(self.aencoder.parameters()) +
            list(self.segmentor.parameters()),
            lr=self.args.lr
        )
        # 负责风格
        self.opt_style = optim.Adam(
            list(self.sencoder.parameters()) +
            list(self.decoder.parameters()),
            lr=self.args.lr
        )
        # 学习率调度器
        self.scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_ae,
            mode='min',
            factor=self.args.lr_decay_rate,
            patience=self.args.lr_patience,
            verbose=True
        )
        self.scheduler_style = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_style,
            mode='min',
            factor=self.args.lr_decay_rate,
            patience=self.args.lr_patience,
            verbose=True
        )

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型参数
        self.aencoder.load_state_dict(checkpoint['aencoder'])
        self.sencoder.load_state_dict(checkpoint['sencoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.segmentor.load_state_dict(checkpoint['segmentor'])

        # 加载优化器状态
        self.opt_ae.load_state_dict(checkpoint['opt_ae'])
        self.opt_style.load_state_dict(checkpoint['opt_style'])

        # 设置开始的epoch
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {self.start_epoch}")

    def train_step(self, data_dict):
        images = data_dict['image'].to(self.device)
        masks = data_dict['mask'].to(self.device)
        print("images.shape", images.shape)  # [18,3,256,256]
        print("masks.shape", masks.shape)  # [18,256,256]
        domain_list = data_dict['domain'].to(self.device)
        # print("domain_list.shape", domain_list.shape)  # [18]
        # print("domain_list:", domain_list)  # [0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

        # 内容编码和分割
        content_features = self.aencoder(images)  # [18,8,256,256]
        seg_pred = self.segmentor(content_features)  # [18,3,256,256]
        print("content_features", content_features.shape)
        print("seg_pred.shape:", seg_pred.shape)
        seg_loss = self.segmentor.Seg_loss(seg_pred, masks)
        # print("seg_loss：",seg_loss)

        # 风格编码
        style_code, mu, logvar = self.sencoder(images)  # [18, 16]
        print("style_code.shape:", style_code.shape)
        kl_loss = self.sencoder.KL_loss(mu, logvar)
        # print("KL_loss", kl_loss)

        # 图片重建
        reconstructed = self.decoder(content_features, style_code)  # [18, 3, 256, 256])
        print("reconstructed_image.shape:", reconstructed.shape)
        rec_loss = self.decoder.reconstructed_loss(images, reconstructed)
        # print("rec_loss", rec_loss)

        # 风格增强损失
        style_aug_loss = StyleAugmentationLoss(content_features, style_code)
        # print("style_aug_loss", style_aug_loss)

        # 对比损失
        contrastive_loss=DomainStyleContrastiveLoss(style_code, domain_list)
        # print("contrastive_loss:", contrastive_loss)

        # 总损失
        total_loss = (
                seg_loss +
                args.lambda_kl * kl_loss +
                args.lambda_rec * rec_loss +
                args.lambda_dsct * contrastive_loss +
                args.lambda_aug * style_aug_loss
        )

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'kl_loss': kl_loss,
            'rec_loss': rec_loss,
            'contrastive_Loss': contrastive_loss,
            'style_aug_loss': style_aug_loss
        }

    def train(self):
        best_loss = float('inf')

        for epoch in range(self.start_epoch, self.args.epoches):
            epoch_loss = 0
            batch_count = 0

            # 创建进度条
            pbar = tqdm(enumerate(self.dataloaders), 
                       total=len(self.dataloaders),
                       desc=f'Epoch {epoch}/{self.args.epoches-1}',
                       ncols=100)

            for i, data in pbar:
                loss_dict = self.train_step(data)
                epoch_loss += loss_dict['total_loss'].item()
                batch_count += 1

                # 优化器更新
                self.opt_ae.zero_grad()
                self.opt_style.zero_grad()
                loss_dict['total_loss'].backward()
                self.opt_ae.step()
                self.opt_style.step()

                # 更新进度条显示的损失值
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'seg_loss': f"{loss_dict['seg_loss']:.4f}",
                    'kl_loss': f"{loss_dict['kl_loss']:.4f}"
                })

                if i % self.args.print_interval == 0:
                    step = epoch * len(self.dataloaders) + i
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v, step)
                    self.writer.add_scalar('lr/ae', self.opt_ae.param_groups[0]['lr'], step)
                    self.writer.add_scalar('lr/style', self.opt_style.param_groups[0]['lr'], step)

            # 计算epoch平均损失
            avg_loss = epoch_loss / batch_count
            print(f"\nEpoch {epoch}, Average Loss: {avg_loss:.4f}")

            # 更新学习率
            self.scheduler_ae.step(avg_loss)
            self.scheduler_style.step(avg_loss)

            # 保存checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch)

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            print(f"Current learning rate: ae={self.opt_ae.param_groups[0]['lr']:.6f}, "
                  f"style={self.opt_style.param_groups[0]['lr']:.6f}")
            print(f"Epochs without improvement: {self.epochs_without_improvement}")

    def save_checkpoint(self, epoch, is_best=False):
        save_dict = {
            'epoch': epoch,
            'aencoder': self.aencoder.state_dict(),
            'sencoder': self.sencoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'segmentor': self.segmentor.state_dict(),
            'opt_ae': self.opt_ae.state_dict(),
            'opt_style': self.opt_style.state_dict(),
            'args': self.args
        }

        # 保存最新的checkpoint
        save_path = os.path.join(
            self.args.checkpoint_dir,
            str(self.args.test_domain[0]),
            f'checkpoint_epoch_{epoch}.pth'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(save_dict, save_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(
                self.args.checkpoint_dir,
                str(self.args.test_domain[0]),
                'model_best.pth'
            )
            torch.save(save_dict, best_path)

        print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()
