import os
from PIL import Image
import numpy as np

def convert_mask(input_path, output_path):
    """转换单个掩码文件的像素值"""
    # 映射规则
    mapping = {
        0: 255,  # 0 -> 255 (白色)
        1: 128,  # 1 -> 128 (灰色)
        2: 0     # 2 -> 0   (黑色)
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

def process_domain(domain_id):
    """处理一个域的所有掩码"""
    # 源掩码目录
    mask_dir = os.path.join('.', str(domain_id), 'mask')
    
    # 目标掩码目录（在域目录下的masks文件夹）
    output_dir = os.path.join('.', str(domain_id), 'masks')
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保源目录存在
    if not os.path.exists(mask_dir):
        print(f"域 {domain_id} 的掩码目录不存在")
        return
    
    # 获取所有掩码文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.bmp'))]
    
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

def main():
    # 处理所有域
    domains = [0, 1, 2, 3]
    for domain_id in domains:
        process_domain(domain_id)
    
    print("\n所有掩码处理完成！转换后的掩码已保存在各个域目录下的 masks 文件夹中")

if __name__ == '__main__':
    main() 