import json
import os
from PIL import Image
import requests
from io import BytesIO

# 创建示例数据
sample_data = [
    {
        "image_name": "sample1.jpg",
        "description": "A white cotton t-shirt with round neck and short sleeves",
        "category": "T-shirts",
        "attributes": {"color": "white", "material": "cotton"}
    },
    {
        "image_name": "sample2.jpg",
        "description": "Black leather jacket with silver zipper details",
        "category": "Jackets",
        "attributes": {"color": "black", "material": "leather"}
    },
    {
        "image_name": "sample3.jpg",
        "description": "Blue denim jeans with straight cut and five pockets",
        "category": "Jeans",
        "attributes": {"color": "blue", "material": "denim"}
    }
]

def create_sample_dataset(output_dir: str = 'data'):
    """创建示例数据集"""
    
    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 保存数据描述
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # 使用占位图片
    for item in sample_data:
        # 创建一个简单的彩色图片
        img = Image.new('RGB', (224, 224), color=item['attributes']['color'])
        img.save(os.path.join(output_dir, 'images', item['image_name']))
    
    print(f"示例数据集创建完成！数据保存在: {output_dir}")

if __name__ == "__main__":
    create_sample_dataset()