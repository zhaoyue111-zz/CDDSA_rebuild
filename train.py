import os
import sys

from torchvision.transforms import transforms

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tensorflow报警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 插件注册
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import argparse
# torch.backends.cudnn.enabled = False # 除非你有特定的原因，否则通常保持cudnn启用
import torch.optim as optim
from models.encoder import AEncoder, SEncoder  # 假设 SEncoder 对应官方 MEncoder
# from models.mydecoder import Decoder  # 假设 Decoder 是你的模型类
from models.mydecoder import Ada_Decoder
from models.segmentor import Segmentor
from data.dataloader import *  # 导入修改后的数据加载器
from torch.utils.data import DataLoader  # 导入 DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.weight_init import initialize_weights  # 导入权重初始化函数
from utils.average_meter import AverageMeter  # 假设你有一个平均值计量器工具类
from datetime import datetime
import random
import numpy as np
from utils.loss import *
from pytorch_metric_learning import losses as pml_losses  # 导入 pytorch_metric_learning 库
from data.dataloader import ToTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--data_dir', type=str, default='/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data')
    parser.add_argument('--img_size', type=int, default=256, help='image size for training')
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--train_domains', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--all_domains', nargs='+', type=int, default=[0, 1, 2,3])
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes for segmentation')
    parser.add_argument('--style_code_dim', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=8, help="the out channel of anatomy encoder")
    # 打印设置
    parser.add_argument('--batch_size', type=int, default=8,
                        help='total batch_size for train (will be divided among domains)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=8,
                        help='epochs without improvement for early stopping')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='interval of saving checkpoints')
    parser.add_argument('--val_interval', type=int, default=5, help='interval of validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints', help='directory to save checkpoints')
    parser.add_argument('--tensorboard_dir', type=str, default='./results/runs', help='directory for tensorboard logs')
    parser.add_argument('--log_file', type=str, default='./results/train.log', help='path to log file')  # 可以写入日志文件
    # 损失权重 (参考官方代码 train_cddsa.py)
    parser.add_argument('--kl_w', type=float, default=0.001, help='weight for KL loss')
    parser.add_argument('--reco_w', type=float, default=0.1, help='weight for reconstruction loss')
    parser.add_argument('--recoz_w', type=float, default=0.1, help='weight for style reconstruction loss')
    parser.add_argument('--style_w', type=float, default=0.1, help='weight for domain style contrastive loss')
    parser.add_argument('--weight_init', type=str, default='kaiming', choices=['kaiming', 'xavier'],
                        help='weight initialization method')

    parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint (for a specific LODO fold)')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # 如果你关心完全确定性，可以启用，但可能略微降低性能
    torch.backends.cudnn.benchmark = False # 如果输入尺寸不变，可以设置为True提高性能


class Trainer:
    def __init__(self, args, val_domain, train_domains):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.val_domain = val_domain  # 当前的测试域ID
        self.train_domains = train_domains  # 当前 LODO 折叠的训练域列表

        # 模型定义
        # 官方 MEncoder 对应 SEncoder (Style Encoder)
        self.sencoder = SEncoder(
            img_size=args.img_size,
            style_code_dim=args.style_code_dim,
            in_channels=args.in_channel,
            out_channels=32,
            hidden_channels=16
        ).to(self.device)

        self.aencoder = AEncoder(
            in_channels=args.in_channel,
            hidden_channels=16,
            out_channels=args.anatomy_channel
        ).to(self.device)

        self.decoder = Ada_Decoder(
            anatomy_out_channel=args.anatomy_channel,
            z_length=args.style_code_dim,
            out_channel=args.in_channel
        ).to(self.device)

        self.segmentor = Segmentor(
            in_channel=args.anatomy_channel,
            hidden_size=32,
            num_classes=args.num_classes
        ).to(self.device)

        # 使用 pytorch_metric_learning 库的 NTXentLoss for 域间风格对比损失
        self.style_contrastive_criterion = pml_losses.NTXentLoss(temperature=0.1).cuda()

        # 优化器
        # 官方代码将 AEncoder 和 Segmentor 放在一个优化器中，SEncoder 和 Decoder 放在另一个优化器中
        self.opt_ae = optim.Adam(list(self.aencoder.parameters()) + list(self.segmentor.parameters()), lr=args.lr)
        self.opt_style = optim.Adam(list(self.sencoder.parameters()) + list(self.decoder.parameters()), lr=args.lr)

        # 学习率调度器
        self.scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_ae, 'min',
                                                                       patience=self.args.early_stopping_patience,
                                                                       factor=args.lr_decay_rate)
        self.scheduler_style = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_style, 'min',
                                                                          patience=self.args.early_stopping_patience,
                                                                          factor=args.lr_decay_rate)

        # 检查点目录
        # 为每个 LODO 折叠创建一个独立的检查点目录
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, f'test_domain_{val_domain}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, f'test_domain_{val_domain}',
                                                         datetime.now().strftime('%Y%m%d_%H%M%S.%f')))

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # 初始化权重或恢复模型
        if args.resume:
            self.load_checkpoint(args.resume)
        else:
            initialize_weights(self.sencoder, args.weight_init)
            initialize_weights(self.aencoder, args.weight_init)
            initialize_weights(self.decoder, args.weight_init)
            initialize_weights(self.segmentor, args.weight_init)

    def load_checkpoint(self, checkpoint_path):
        print(f"Attempting to load checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting training from scratch.")
            return

        checkpoint = torch.load(checkpoint_path)
        self.aencoder.load_state_dict(checkpoint['aencoder'])
        self.sencoder.load_state_dict(checkpoint['sencoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.segmentor.load_state_dict(checkpoint['segmentor'])
        self.opt_ae.load_state_dict(checkpoint['opt_ae'])
        self.opt_style.load_state_dict(checkpoint['opt_style'])

        # 如果 checkpoint 中有这些信息，则恢复它们
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        # if 'epochs_without_improvement' in checkpoint:
        #     self.epochs_without_improvement = checkpoint['epochs_without_improvement']

        print(f"Successfully loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, epoch, is_best=False):
        save_dict = {
            'epoch': epoch,
            'aencoder': self.aencoder.state_dict(),
            'sencoder': self.sencoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'segmentor': self.segmentor.state_dict(),
            'opt_ae': self.opt_ae.state_dict(),
            'opt_style': self.opt_style.state_dict(),
            'args': self.args,
            'best_val_loss': self.best_val_loss
        }

        # 保存最新的 checkpoint
        save_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(save_dict, save_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(save_dict, best_path)
            print(f"Saved best model for test domain {self.val_domain} to {best_path}")


    def train_epoch(self, train_iterator, epoch):
        self.aencoder.train()
        self.sencoder.train()
        self.decoder.train()
        self.segmentor.train()

        total_losses_meter = AverageMeter()
        inner_losses_meter = AverageMeter()  # 总损失
        seg_losses_meter = AverageMeter()
        reco_losses_meter = AverageMeter()
        recoz_losses_meter = AverageMeter()
        kl_losses_meter = AverageMeter()
        style_losses_meter = AverageMeter()

        # 使用 MultiDomainIterator 迭代，每次 `sample` 是一个列表
        # 列表中每个元素是一个域的 mini-batch
        for batch_idx, sample_list_per_domain in enumerate(train_iterator):
            # 一个域
            self.opt_ae.zero_grad()
            self.opt_style.zero_grad()

            # 收集当前整个 mini-batch 中所有域的风格和内容特征，用于域间损失计算
            all_style_features_in_batch = []
            all_domain_labels_in_batch = []  # 用于风格对比损失的域标签

            total_intra_domain_loss_in_batch = 0  # 累加当前 mini-batch 中所有域的域内损失

            dc_num = len(self.train_domains)  # 训练域的数量

            # --- 域内损失计算 (Intra-domain Losses) ---
            for dc, domain_batch in enumerate(sample_list_per_domain):
                image = domain_batch['image'].cuda()
                gt = domain_batch['mask'].cuda()
                domain_id_tensor = domain_batch['domain_id'].cuda()  # 域ID

                # 前向传播
                a_out = self.aencoder(image)
                # print("a_out", a_out.shape)  # [B,8,256,256]
                # 1. 分割损失
                seg_pred = self.segmentor(a_out)
                # print("seg_pred",seg_pred.shape)  # [B,3,256,256]
                seg_loss = Seg_loss(seg_pred, gt)  # BCEWithLogitsLoss 需要 float target

                z_out, mu_out, logvar_out = self.sencoder(image)  # 风格编码器
                # print("z_out",z_out.shape)  # [B,16]
                # 2. KL散度损失 (正则化风格隐空间)
                kl_loss = KL_loss(logvar_out, mu_out)

                # 3. 重建损失
                rec = self.decoder(a_out, z_out)
                # print("rec:",rec.shape) # [B,3,256,256]
                rec_loss = Reconstructed_loss(image,rec)

                # 4. 风格重建损失
                recoz_loss=StyleAugmentationLoss(a_out,z_out,self.decoder,self.aencoder)

                # 每个域的域内损失 (平均每个样本的损失)
                domain_current_intra_loss = seg_loss + \
                                            self.args.kl_w * kl_loss + \
                                            self.args.reco_w * rec_loss + \
                                            self.args.recoz_w * recoz_loss

                total_intra_domain_loss_in_batch += domain_current_intra_loss

                # 收集当前域的风格和内容特征，用于后续的域间损失计算
                all_style_features_in_batch.append(z_out)
                all_domain_labels_in_batch.append(domain_id_tensor)

                # 记录域内损失
                # 注意：meter update 的 `n` 参数应是当前子批次的大小
                inner_losses_meter.update(domain_current_intra_loss.item(), image.size(0))
                seg_losses_meter.update(seg_loss.item(), image.size(0))
                reco_losses_meter.update(rec_loss.item(), image.size(0))
                recoz_losses_meter.update(recoz_loss.item(), image.size(0))
                kl_losses_meter.update(kl_loss.item(), image.size(0))

            # --- 域间损失计算 (Inter-domain Losses) ---
            # 在处理完当前 mini-batch 中的所有域数据后计算
            # 拼接所有域的风格特征和域标签 (整个 batch 的)
            all_style_features_in_batch_concat = torch.cat(all_style_features_in_batch, dim=0)
            all_domain_labels_in_batch_concat = torch.cat(all_domain_labels_in_batch, dim=0)

            # 域间风格对比损失 (Domain Style Contrastive Loss)
            # 直接使用 pytorch_metric_learning 的 NTXentLoss  理想状态style_loss会非常小，因此后面直接加
            style_loss = self.style_contrastive_criterion(all_style_features_in_batch_concat,
                                                          all_domain_labels_in_batch_concat)
            style_losses_meter.update(style_loss.item(), all_style_features_in_batch_concat.size(0))

            # --- 总损失 (Total Loss) ---
            # 将所有域的域内损失平均，再加上加权的域间损失
            total_loss = (total_intra_domain_loss_in_batch / dc_num) + \
                         self.args.style_w * style_loss

            total_losses_meter.update(total_loss.item(), all_domain_labels_in_batch_concat.size(0))

            # 反向传播和优化
            total_loss.backward()
            self.opt_ae.step()
            self.opt_style.step()

            # 记录到 TensorBoard (每个 batch 记录一次，如果数据量大，可以调整为每个 epoch 记录)
            global_step = epoch * len(train_iterator) + batch_idx
            self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
            self.writer.add_scalar('Train/Seg_Loss', seg_losses_meter.avg, global_step)
            self.writer.add_scalar('Train/Reco_Loss', reco_losses_meter.avg, global_step)
            self.writer.add_scalar('Train/RecoZ_Loss', recoz_losses_meter.avg, global_step)
            self.writer.add_scalar('Train/KL_Loss', kl_losses_meter.avg, global_step)
            self.writer.add_scalar('Train/Style_Loss', style_losses_meter.avg, global_step)

        print(
            f"Epoch {epoch} Train Loss: {total_losses_meter.avg:.4f} (total), "
            f"Seg: {seg_losses_meter.avg:.4f}, "
            f"Reco: {reco_losses_meter.avg:.4f}, "
            f"RecoZ: {recoz_losses_meter.avg:.4f}, "
            f"KL: {kl_losses_meter.avg:.4f}, "
            f"Style: {style_losses_meter.avg:.4f}")

        return total_losses_meter.avg

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.aencoder.eval()
        self.sencoder.eval()
        self.decoder.eval()
        self.segmentor.eval()

        val_losses_meter = AverageMeter()  # 总验证损失
        val_seg_losses_meter = AverageMeter()
        val_reco_losses_meter = AverageMeter()

        for batch_idx, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            mask = sample['mask'].cuda()

            # 前向传播
            a_out = self.aencoder(image)
            z_out, mu_out, logvar_out = self.sencoder(image)

            # 仅计算分割损失和重建损失用于验证
            seg_pred = self.segmentor(a_out)
            seg_loss = Seg_loss(seg_pred, mask)

            reco = self.decoder(a_out, z_out)
            reco_loss = Reconstructed_loss(image,reco)

            # 验证总损失 (可以根据论文或实际需求调整权重，通常侧重于分割性能)
            val_loss = seg_loss + self.args.reco_w * reco_loss

            val_losses_meter.update(val_loss.item(), image.size(0))
            val_seg_losses_meter.update(seg_loss.item(), image.size(0))
            val_reco_losses_meter.update(reco_loss.item(), image.size(0))

        print(
            f"Epoch {epoch} Val Loss: {val_losses_meter.avg:.4f}, Val Seg: {val_seg_losses_meter.avg:.4f}, Val Reco: {val_reco_losses_meter.avg:.4f}")

        self.writer.add_scalar('Val/Total_Loss', val_losses_meter.avg, epoch)
        self.writer.add_scalar('Val/Seg_Loss', val_seg_losses_meter.avg, epoch)
        self.writer.add_scalar('Val/Reco_Loss', val_reco_losses_meter.avg, epoch)
        return val_losses_meter.avg

    def run(self, train_iterator, val_loader):
        for epoch in range(1, self.args.epochs + 1):
            print("Epoch:",epoch)
            train_loss = self.train_epoch(train_iterator, epoch)

            if epoch % self.args.val_interval == 0:
                val_loss = self.validate(val_loader, epoch)
                self.scheduler_ae.step(val_loss)
                self.scheduler_style.step(val_loss)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                self.save_checkpoint(epoch, is_best)

                print(f"Current learning rate: ae={self.opt_ae.param_groups[0]['lr']:.6f}, "
                      f"style={self.opt_style.param_groups[0]['lr']:.6f}")
                print(f"Epochs without improvement: {self.epochs_without_improvement}")

                if self.epochs_without_improvement >= self.args.early_stopping_patience:
                    print(
                        f"Early stopping triggered after {self.args.early_stopping_patience} epochs without improvement for test domain {self.current_test_domain_id}.")
                    break


def main():
    args = parse_args()
    set_seed(args.seed)  # 设置随机种子

    # --- Leave-One-Domain Out (LODO) 循环 ---
    all_train_domains = args.train_domains # [1 2 3]

    # 用于记录每个测试域的结果
    lodo_results = {}

    for val_domain_id in all_train_domains:  # [1,2,3]
        train_domains = [d for d in all_train_domains if d != val_domain_id]  # 1->2 3

        if not train_domains:
            print(f"Skipping LODO fold for test domain {val_domain_id}: No training domains specified.")
            continue

        print(f"\n--- Starting LODO Fold: Test Domain = {val_domain_id} ---")  # 1
        print(f"Train Domains: {train_domains}")  # 2 3

        # 数据加载
        # `batch_size_per_domain` 是每个域的子批次大小
        # todo 确保 `args.batch_size` 能被 `len(train_domains)` 整除，以避免最后一个 batch 不完整
        if args.batch_size % len(train_domains) != 0:
            print(
                f"Warning: args.batch_size ({args.batch_size}) is not divisible by number of train domains ({len(train_domains)}).")
            print(
                f"Adjusting batch_size_per_domain to {args.batch_size // len(train_domains)}. Last samples in domains might be dropped.")
        batch_size_per_domain = args.batch_size // len(train_domains)
        if batch_size_per_domain == 0:
            raise ValueError(
                f"Calculated batch_size_per_domain is 0. Please increase --batch_size or decrease number of train domains. "
                f"Current batch_size: {args.batch_size}, Number of train domains: {len(train_domains)}")

        # 训练数据迭代器
        composed_transforms = transforms.Compose([
            ToTensor()
        ])
        train_iterator_lodo = MultiDomainIterator(
            root_dir=args.data_dir,
            domain_list=train_domains,         # [2,3]
            batch_size_per_domain=batch_size_per_domain,
            transform=composed_transforms
        )

        # 验证数据加载器 (只包含当前测试域)
        val_dataset_lodo = FundusDataset(
            root_dir=args.data_dir,
            domain=val_domain_id,
            transform=composed_transforms
        )
        val_loader_lodo = DataLoader(val_dataset_lodo, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                     drop_last=False) # 所有数据都要取，最后一个batch作为一个很小的数据

        # 创建 Trainer 实例
        trainer = Trainer(args, val_domain_id, train_domains)  # 传入 train_domains 给 Trainer
        trainer.run(train_iterator_lodo, val_loader_lodo)

        # 记录当前 LODO 折叠的结果
        lodo_results[val_domain_id] = trainer.best_val_loss
        print(f"Finished LODO Fold for Test Domain {val_domain_id}. Best Val Loss: {trainer.best_val_loss:.4f}")

        # 清理 GPU 内存，避免交叉验证过程中 OOM
        del trainer
        torch.cuda.empty_cache()

    print("\n--- All LODO Folds Completed ---")
    print("LODO Results (Test Domain ID: Best Validation Loss):")
    for domain_id, loss in lodo_results.items():
        print(f"  Domain {domain_id}: {loss:.4f}")

    # 计算平均结果
    if lodo_results:
        avg_loss = sum(lodo_results.values()) / len(lodo_results)
        print(f"Average Best Validation Loss across all LODO Folds: {avg_loss:.4f}")
    else:
        print("No LODO folds were completed.")


if __name__ == '__main__':
    main()