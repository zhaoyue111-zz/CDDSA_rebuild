import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms
import random

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        # Convert PIL Image to Tensor, handle grayscale masks for segmentation
        image = transforms.ToTensor()(image)
        # For mask, convert to LongTensor and ensure it's single channel (H, W)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return {'image': image, 'mask': mask, 'domain_id': sample['domain_id']}

class FundusDataset(Dataset):
    '''
        获取一个特定域的所有图片、标签、域号
    '''
    def __init__(self, root_dir="./OD", domain=0, transform=None):
        self.root_dir = os.path.join(root_dir, str(domain))
        self.domain = domain
        self.transform = transform

        self.image_dir = os.path.join(self.root_dir, 'image')
        self.mask_dir = os.path.join(self.root_dir, 'mask')

        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])
        # 检查是否有文件
        if not self.image_files:
            raise ValueError(f"No image files found in {self.image_dir}. Please check the path and file extensions.")

    def __len__(self):
        return len(self.image_files)

    # 获取一个batch中的每个图片
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        sample = {'image': image, 'mask': mask, 'domain_id': self.domain}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MultiDomainIterator:
    '''
        一个迭代器，每次迭代返回一个列表，其中每个元素是一个特定域的mini-batch。
        模仿官方代码中 `sample_minibatch_fundus` 的行为。
        确保每个返回的“mini-batch of domains”中包含所有指定域的子批次。
    '''
    def __init__(self, root_dir, domain_list, batch_size_per_domain, transform=None):
        self.datasets = {}  # 包含domain_list中所有域的batch_size_per_domain个数据
        self.data_loaders = {}
        self.iterators = {}
        self.domain_list = domain_list  # 要训练的域列表
        self.batch_size_per_domain = batch_size_per_domain
        self.current_iteration = 0

        for domain_id in domain_list:
            dataset = FundusDataset(root_dir=root_dir, domain=domain_id,transform=transform)
            # DataLoader的batch_size设置为每个域的batch_size_per_domain
            # drop_last=True 确保每个域的子批次大小一致
            dataloader = DataLoader(dataset, batch_size=batch_size_per_domain, shuffle=True, num_workers=0,
                                    drop_last=True)
            self.datasets[domain_id] = dataset
            self.data_loaders[domain_id] = dataloader
            self.iterators[domain_id] = iter(dataloader)

        # 最大迭代次数取决于最短的dataloader
        # 如果某个域的DataLoader为空，则min()会报错，需要处理
        if not self.data_loaders:
            self.max_iterations = 0
        else:
            self.max_iterations = min(len(dl) for dl in self.data_loaders.values())

    def __len__(self):
        return self.max_iterations

    # 一次迭代取一个batch数据
    def __iter__(self):
        for domain_id in self.domain_list:
            self.iterators[domain_id] = iter(self.data_loaders[domain_id])
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.max_iterations:
            raise StopIteration

        batch_per_domain = []
        for domain_id in self.domain_list:
            batch_per_domain.append(next(self.iterators[domain_id]))

        self.current_iteration += 1
        return batch_per_domain  # 返回一个列表，每个元素是一个域的字典batch


# --- MultiDomainDataset (用于验证集，因为它只包含一个域)
class MultiDomainDataset(Dataset):
    '''
        用于组合来自多个域的数据集，并为DataLoader提供一个统一的接口。
        这样DataLoader会返回一个混合了所有指定域数据的batch。
        用于验证集时，split_ids通常只包含一个域。
    '''

    def __init__(self, root_dir, split_ids, transform=None):
        self.datasets = []
        self.domain_map = {}  # 记录原始 domain_id 到内部索引的映射
        self.domain_start_indices = []  # 记录每个域数据在合并后的总列表中的起始索引

        current_idx = 0
        for i, domain_id in enumerate(split_ids):
            dataset = FundusDataset(root_dir=root_dir, domain=domain_id, transform=transform)
            self.datasets.append(dataset)
            self.domain_map[domain_id] = i  # 存储原始domain_id及其内部索引
            self.domain_start_indices.append(current_idx)
            current_idx += len(dataset)

        self.total_len = current_idx  # 合并后的总长度

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 根据图片idx找到对应的原始域和该域内的索引
        '''
            比如0:200 1:300 2:400  domain_start_indices:0,200,500,900
            idx=315 取自1域，在1域的索引是315-200=115
        '''
        domain_idx = 0
        for i, start_idx in enumerate(self.domain_start_indices):
            if idx >= start_idx:  # 找到最后一个
                domain_idx = i  # 315->1
            else:
                break

        # 计算该域内的相对索引
        relative_idx = idx - self.domain_start_indices[domain_idx]  # 315-200=115
        # 取样本
        sample = self.datasets[domain_idx][relative_idx]
        return sample


if __name__ == '__main__':
    root_dir = '/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data'

    composed_transforms = transforms.Compose([
        ToTensor()
    ])

    test_domain_id = 0
    train_domains =[1,2]

    print(f"Train Domains: {train_domains}")
    print(f"Test Domain: {test_domain_id}")

    # 训练数据迭代器 (使用 MultiDomainIterator)
    total_batch_size = 8
    batch_size_per_domain = total_batch_size // len(train_domains)

    train_iterator = MultiDomainIterator(
        root_dir,
        domain_list=train_domains,
        batch_size_per_domain=batch_size_per_domain,
        transform=composed_transforms
    )

    print(f"Train MultiDomainIterator length: {len(train_iterator)}")  # min(400,650,1020)/5 最小迭代数

    for i, domains_batch_list in enumerate(train_iterator):
        print(f"\nBatch {i + 1}:")
        # domains_batch_list 是一个列表，每个元素是一个域的 batch dict
        for dc, domain_batch in enumerate(domains_batch_list):
            print(f"  Domain {domain_batch['domain_id'][0].item()} (index {dc}):")
            print(f"    Images shape: {domain_batch['image'].shape}")  #
            print(f"    Masks shape: {domain_batch['mask'].shape}")  #
            print(f"    Domain IDs: {domain_batch['domain_id']}")  #

        if i == 1:  # 只打印几个批次
            break

    # 验证数据加载 (使用 MultiDomainDataset 和 DataLoader)
    val_dataset = MultiDomainDataset(root_dir, split_ids=[test_domain_id], transform=composed_transforms)
    val_loader = DataLoader(val_dataset, batch_size=total_batch_size, shuffle=False, num_workers=0, drop_last=True)

    print(f"\nValidation DataLoader length: {len(val_loader)}")
    for i, batch in enumerate(val_loader):
        print(f"\nValidation Batch {i + 1}:")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Masks shape: {batch['mask'].shape}")
        print(f"Domain IDs: {batch['domain_id']}")
        if i == 1:
            break