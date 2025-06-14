import os
import pandas as pd
import re

def natural_sort_key(s):
    # 将字符串中的数字部分转换为整数进行比较
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_labels_csv(image_dir, output_csv):
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
    # 使用自然排序
    image_files.sort(key=natural_sort_key)
    
    # 创建数据列表
    data = []
    for idx, img_file in enumerate(image_files):
        # 根据索引确定剩余循环数和剥落面积
        if idx < 306:
            remaining_cycles = 56
            spalling_area = 0.19
        elif idx < 612:
            remaining_cycles = 65
            spalling_area = 0.15
        elif idx < 918:
            remaining_cycles = 94
            spalling_area = 0.1834
        elif idx < 1224:
            remaining_cycles = 104
            spalling_area = 0.21203
        elif idx < 1530:
            remaining_cycles = 111
            spalling_area = 0.17632
        
        data.append({
            'image_name': img_file,  # 直接使用原始文件名
            'remaining_cycles': remaining_cycles,
            'spalling_area': spalling_area
        })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    
    # 保存CSV文件，确保没有额外的空行
    df.to_csv(output_csv, index=False)
    
    # 打印统计信息
    print(f"总共处理了 {len(data)} 张图片")
    print("\n每个类别的图片数量：")
    print(df['remaining_cycles'].value_counts().sort_index())
    
    # 检查是否所有图片都存在
    missing_files = [f for f in df['image_name'] if not os.path.exists(os.path.join(image_dir, f))]
    if missing_files:
        print("\n警告：以下图片文件不存在：")
        for f in missing_files:
            print(f)

if __name__ == "__main__":
    # 设置输入输出路径
    image_dir = "./thelast"  # 图片目录
    output_csv = "./data/labels.csv"  # 输出CSV文件路径
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 创建标签文件
    create_labels_csv(image_dir, output_csv)
    
    # 验证生成的CSV文件
    df = pd.read_csv(output_csv)
    print("\n生成的CSV文件前5行：")
    print(df.head())