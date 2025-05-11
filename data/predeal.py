# resize  rename  改后缀
import os
import cv2
from PIL import Image
import numpy as np
from pathlib import Path

def process_domain(domain_id, target_size=256):
    base_dir = '.'
    domain_dir = os.path.join(base_dir, str(domain_id))
    image_dir = os.path.join(domain_dir, 'image')
    mask_dir = os.path.join(domain_dir, 'mask')

    # 创建临时目录
    temp_image_dir = os.path.join(domain_dir, 'image_processed')
    temp_mask_dir = os.path.join(domain_dir, 'mask_processed')
    os.makedirs(temp_image_dir, exist_ok=True)
    os.makedirs(temp_mask_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # 确定mask文件的扩展名
    mask_ext = '.bmp' if domain_id in [0, 2] else '.png'

    for idx, img_file in enumerate(sorted(image_files)):
        # 构建新的文件名
        new_name = f"{idx:04d}"

        # 处理图像
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片：{img_path}")
            continue

        # resize图像
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 保存处理后的图像
        new_img_path = os.path.join(temp_image_dir, f"{new_name}.jpg")
        cv2.imwrite(new_img_path, img_resized)

        # 处理对应的mask
        mask_file = os.path.splitext(img_file)[0] + mask_ext
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"未找到对应的mask文件：{mask_path}")
            continue

        # 读取mask（使用PIL以确保正确读取不同格式）
        mask = np.array(Image.open(mask_path))

        # resize mask（使用最近邻插值以保持标签值）
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 保存处理后的mask（统一为PNG格式）
        new_mask_path = os.path.join(temp_mask_dir, f"{new_name}.png")
        cv2.imwrite(new_mask_path, mask_resized)

        print(f"处理完成 {domain_id}/{img_file} -> {new_name}")

    # 处理完成后，替换原始目录
    # import shutil
    # shutil.rmtree(image_dir)
    # shutil.rmtree(mask_dir)
    # os.rename(temp_image_dir, image_dir)
    # os.rename(temp_mask_dir, mask_dir)

    print(f"域 {domain_id} 处理完成")


def convert_mask(input_path, output_path):
    """转换单个掩码文件的像素值"""
    # 映射规则
    mapping = {
        255: 0,  # 0 -> 255 (白色)
        128: 1,  # 1 -> 128 (灰色)
        0: 2  # 2 -> 0   (黑色)
    }

    # 读取掩码
    mask = np.array(Image.open(input_path))

    # 创建新的掩码数组
    new_mask = np.zeros_like(mask)

    # 应用映射
    for old_value, new_value in mapping.items():
        new_mask[mask == old_value] = new_value

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存转换后的掩码
    Image.fromarray(new_mask.astype(np.uint8)).save(output_path)


def process_pixel(domain_id):
    """处理一个域的所有掩码"""
    # 源掩码目录
    mask_dir = os.path.join('.', str(domain_id), 'mask')

    # 目标掩码目录
    output_dir = os.path.join('.', str(domain_id),'masks')
    os.makedirs(output_dir, exist_ok=True)

    # 确保源目录存在
    if not os.path.exists(mask_dir):
        print(f"域 {domain_id} 的掩码目录不存在")
        return

    # 获取所有掩码文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png'))]

    print(f"开始处理域 {domain_id} 的掩码...")
    for mask_file in mask_files:
        input_path = os.path.join(mask_dir, mask_file)
        # 统一输出为png格式
        output_filename = os.path.splitext(mask_file)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)

        try:
            convert_mask(input_path, output_path)
            print(f"已处理: {mask_file} -> {output_filename}")
        except Exception as e:
            print(f"处理 {mask_file} 时出错: {str(e)}")

    print(f"域 {domain_id} 的掩码处理完成，保存在 {output_dir}")

def process_all_domains():
    for domain_id in [0,1,2,3]:
        print(f"开始处理域 {domain_id}")
        process_pixel(domain_id)
        print(f"域 {domain_id} 处理完成\\n")

if __name__ == '__main__':
    process_all_domains()
