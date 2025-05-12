'''
底为白，中间为黑，稍大是灰
0-ok
2-ok
1-
3-{0: 255,1: 128,2: 0} ok
'''

import os
from PIL import Image, ImageDraw
import numpy as np
import cv2

def dealDomain1():
    # 设置输入输出文件夹路径
    input_folder = "/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/0/gt"  # 替换为你的输入文件夹路径
    output_folder = "/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/0/mask"  # 替换为你的输出文件夹路径

    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 映射规则
    mapping = {
        255:0,   # 0-视盘
        128:1,   # 1-视杯
        0: 2    # 2-背景
    }

    # 遍历文件夹内所有PNG图像
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 加载图像并转灰度
            img = Image.open(input_path).convert("L")
            arr = np.array(img)
            print("图像中唯一像素值: ", np.unique(arr))

            # 应用像素映射
            mapped = np.vectorize(mapping.get)(arr)
            result_img = Image.fromarray(mapped.astype(np.uint8))

            # 保存处理后的图像
            result_img.save(output_path)

            print(f"已处理: {filename}")

dealDomain1()

def read_coordinates(txt_file):
    with open(txt_file, "r") as f:
        # 读取 (x, y)
        points = [list(map(int, line.strip().split())) for line in f if line.strip()]
    # 转换为 (y, x) → OpenCV 格式 (row, col)
    points = [[y, x] for x, y in points]

    # 转换为 np.array + reshape 成 (N, 1, 2)
    arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # 打印检查：类型、形状、示例
    print("坐标 array 类型：", type(arr))
    print("array dtype：", arr.dtype)
    print("array shape：", arr.shape)
    print("示例前2点：", arr[:2])

    return arr

def dealDomain2():
    print(1)
    # 输入边界文件所在根目录
    boundary_dir = "./2/test/Test/Test_GT"
    # 输出掩码保存目录
    output_dir = "./2/mask"
    os.makedirs(output_dir, exist_ok=True)

    # 图像尺寸（请根据实际图像调整）
    width, height = 2049, 1751

    for case_name in os.listdir(boundary_dir):
        case_path = os.path.join(boundary_dir, case_name, "AvgBoundary")
        if not os.path.isdir(case_path):
            continue

        cup_file = os.path.join(case_path, f"{case_name}_CupAvgBoundary.txt")
        od_file = os.path.join(case_path, f"{case_name}_ODAvgBoundary.txt")

        if os.path.exists(cup_file) and os.path.exists(od_file):
            mask = np.zeros((height, width), dtype=np.uint8)
            od_points = read_coordinates(od_file)
            cup_points = read_coordinates(cup_file)

            # 先画 OD 再画 Cup
            cv2.fillPoly(mask, [od_points], 128)
            cv2.fillPoly(mask, [cup_points], 255)

            save_path = os.path.join(output_dir, f"{case_name}_mask.png")
            cv2.imwrite(save_path, mask)
            print(f"已生成: {save_path}")


import numpy as np
import cv2
from PIL import Image


def get_bounding_box_from_mask(mask, target_value):
    # 通过掩码图像中的目标值，找到视盘区域的边界框
    coords = np.column_stack(np.where(mask == target_value))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


def crop_to_disk(image_path, mask_path, target_value, out_img_path, out_mask_path):
    # 读取图像和掩码
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取图片：{image_path}")
        return
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[错误] 无法读取掩码：{mask_path}")
        return
    
    # 添加调试信息
    unique_values = np.unique(mask)
    print(f"掩码中的唯一值: {unique_values}")
    print(f"目标值: {target_value}")
    
    # 获取视盘坐标
    coords = np.column_stack(np.where(mask == target_value))
    if coords.size == 0:
        print(f"[跳过] 未找到视盘区域：{image_path}")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # 目标大小
    crop_size = 512
    half = crop_size // 2

    # 计算裁剪范围
    h, w = img.shape[:2]
    top = max(0, center_y - half)
    bottom = min(h, center_y + half)
    left = max(0, center_x - half)
    right = min(w, center_x + half)

    # 裁剪
    cropped = img[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]

    # 若尺寸小于512，则填充
    pad_top = max(0, half - center_y)
    pad_bottom = max(0, (center_y + half) - h)
    pad_left = max(0, half - center_x)
    pad_right = max(0, (center_x + half) - w)

    cropped_padded = cv2.copyMakeBorder(
        cropped, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cropped_mask = cv2.copyMakeBorder(
        cropped_mask, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    # 保存图片（jpg）
    cv2.imwrite(out_img_path, cropped_padded, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # 保存掩码（bmp）
    cv2.imwrite(out_mask_path, cropped_mask)

    print(f"[成功] 处理完成：{image_path}")
    print(f"裁剪尺寸: {cropped.shape}")
    print(f"填充后尺寸: {cropped_padded.shape}")

def batch_process(img_dir, mask_dir, out_img_dir, out_mask_dir, target_value=128):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 获取所有图片文件
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"找到{len(img_names)}张图片")
    
    for img_name in img_names:
        # 获取不带后缀的文件名
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(img_dir, f"{base_name}.jpg")
        mask_path = os.path.join(mask_dir, f"{base_name}.bmp")
        
        if not os.path.exists(mask_path):
            print(f"[警告] 未找到对应的掩码文件：{mask_path}")
            continue
            
        out_img_path = os.path.join(out_img_dir, f"{base_name}.jpg")
        out_mask_path = os.path.join(out_mask_dir, f"{base_name}.bmp")
        
        print(f"\n处理：{base_name}")
        print(f"图片路径：{img_path}")
        print(f"掩码路径：{mask_path}")
        
        crop_to_disk(img_path, mask_path, target_value, out_img_path, out_mask_path)

def get_bounding_box_size(mask, target_value):
    coords = np.column_stack(np.where(mask == target_value))
    if coords.size == 0:
        return 0, 0  # 没有该标签
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return width, height

def find_max_od_size(mask_folder, target_value=128,fix='.jpg'):
    max_w, max_h = 0, 0
    for fname in os.listdir(mask_folder):
        if not fname.endswith(fix):
            continue
        path = os.path.join(mask_folder, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        w, h = get_bounding_box_size(mask, target_value)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h

# mask_folder = "/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/2/mask"  # 替换为你自己的掩码文件夹路径
# max_width, max_height = find_max_od_size(mask_folder,128,'.bmp')
# print(f"视盘区域的最大宽度为: {max_width}, 最大高度为: {max_height}")

# img_dir="/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/2/image"
# mask_dir="/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/2/mask"
# out_img_dir="/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/2/cropImage"
# out_mask_dir="/IMBR_Data/Student-home/2022U_ZhaoYue/MyCode/CDDSA/data/2/cropMask"
# batch_process(img_dir, mask_dir, out_img_dir, out_mask_dir)