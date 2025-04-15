import os
import json
import requests
from tqdm import tqdm
import pandas as pd
import zipfile
import shutil

def download_file(url: str, filename: str):
    """下载文件，显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def prepare_dataset(output_dir: str = 'data', sample_size: int = 1000):
    """准备Fashion-Gen数据集的小样本"""
    
    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 下载Fashion-Gen数据集（这里使用样例URL，实际使用时需要替换）
    DATASET_URL = "https://fashion-gen.s3.amazonaws.com/fashion-gen.zip"
    zip_path = os.path.join(output_dir, 'fashion-gen.zip')
    
    print("下载数据集...")
    try:
        download_file(DATASET_URL, zip_path)
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载数据集并放置在 data/fashion-gen.zip")
        return
    
    # 解压数据集
    print("解压数据集...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # 读取并处理数据
    print("处理数据...")
    df = pd.read_json(os.path.join(output_dir, 'fashion-gen/dataset.json'))
    
    # 随机抽样
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # 保存处理后的数据
    sample_data = sample_df.to_dict('records')
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # 复制所需的图片
    print("复制图片文件...")
    for item in tqdm(sample_data):
        src_path = os.path.join(output_dir, 'fashion-gen/images', item['image_name'])
        dst_path = os.path.join(output_dir, 'images', item['image_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 清理临时文件
    print("清理临时文件...")
    os.remove(zip_path)
    shutil.rmtree(os.path.join(output_dir, 'fashion-gen'))
    
    print(f"数据集准备完成！共处理 {len(sample_data)} 条数据")
    print(f"数据保存在: {output_dir}")

if __name__ == "__main__":
    prepare_dataset()