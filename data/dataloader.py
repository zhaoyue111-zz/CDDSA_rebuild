import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class FundusDataset(Dataset):
    '''
        获取一个特定域的所有图片、标签、域号
    '''
    def __init__(self, root_dir="./OD", domain=0, transform=None):
        self.root_dir = os.path.join(root_dir, str(domain))
        self.domain = domain
        self.transform = transform

        # 获取图像和分割标签路径
        self.image_dir = os.path.join(self.root_dir, 'image')
        self.mask_dir = os.path.join(self.root_dir, 'mask')

        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                 if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        # 读取图像和mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 转换为tensor
        image = torch.FloatTensor(np.array(image).transpose(2, 0, 1)) / 255.0  # [3, H, W]
        mask = torch.LongTensor(np.array(mask))  # [H, W]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'mask': mask,
            'domain': int(self.domain)
        }

class MultiDomainIterator:
    def __init__(self, root_dir, domain_list,sample_list):
        '''
        domain_list:[0,2,3]
        sample_list:[3,4,5]
        '''
        self.domain_list = domain_list
        self.sample_list = sample_list

        self.datasets = {
            domain: FundusDataset(root_dir, domain)
            for domain in domain_list
        }
        self.domain_sizes = {
            domain: len(dataset)
            for domain, dataset in self.datasets.items()
        }
        # 随机索引
        self.domain_indices = {
            domain: torch.randperm(size).tolist()
            for domain, size in self.domain_sizes.items()
        }
        self.current_indices = {domain: 0 for domain in self.domain_list}

    # 每个epoch都要打乱
    def __iter__(self):
        self.current_indices = {domain: 0 for domain in self.domain_list}
        for domain in self.domain_list:
            self.domain_indices[domain] = torch.randperm(self.domain_sizes[domain]).tolist()
        return self

    def __len__(self):
        "计算一个epoch中的batch数量"
        # 找到限制batch数量的域（样本数/每个batch取样数 最小的那个）
        min_batches = float('inf')
        for domain_idx, domain in enumerate(self.domain_list):
            num_samples = self.domain_sizes[domain]  # 该域的所有图片
            samples_per_batch = self.sample_list[domain_idx]
            possible_batches = num_samples // samples_per_batch
            min_batches = min(min_batches, possible_batches)
        return min_batches

    def __next__(self):
        batch_data = {
            'image': [],
            'mask': [],
            'domain': []
        }
        # 只要有一个域到达末尾就停止迭代
        if any(self.current_indices[domain] >= self.domain_sizes[domain] for domain in self.domain_list):
            raise StopIteration

        for domain_idx,domain in enumerate(self.domain_list):
            # if self.current_indices[domain] >= self.domain_sizes[domain]:
            #     self.domain_indices[domain] = torch.randperm(self.domain_sizes[domain]).tolist()
            #     self.current_indices[domain] = 0

            samples=self.sample_list[domain_idx]
            for i in range(samples):
                idx=self.domain_indices[domain][self.current_indices[domain]]
                data=self.datasets[domain][idx]
                batch_data['image'].append(data['image'].unsqueeze(0))
                batch_data['mask'].append(data['mask'].unsqueeze(0))
                batch_data['domain'].append(torch.tensor([data['domain']]))
                self.current_indices[domain]+=1

        return {
            'image': torch.cat(batch_data['image'], dim=0),
            'mask': torch.cat(batch_data['mask'], dim=0),
            'domain': torch.cat(batch_data['domain'], dim=0)
        }


if __name__ == '__main__':
    root_dir = '/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data'
    domain_list = [0,2,3]
    sample_list=[4,4,10]

    iterator = MultiDomainIterator(root_dir, domain_list, sample_list)
    print("iterator length: {}".format(len(iterator)))

    print("dataloader")
    for i, batch in enumerate(iterator):
        print(f"\\nBatch {i+1}:")
        print(f"Images: {batch['image'].shape}")
        print(f"Domains: {batch['domain']}")

        # 显示每个域的样本数量
        unique_domains, counts = torch.unique(batch['domain'], return_counts=True)
        print("Domain counts:")
        for d, c in zip(unique_domains.tolist(), counts.tolist()):
            print(f"Domain {d}: {c}")
