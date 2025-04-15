import json
import torch
import os
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor  # 更新为新的处理器

from ..models.image_encoder import ImageEncoder
from ..models.text_encoder import TextEncoder
from ..models.fusion import LateFusion
from .vector_store import VectorStore

class DataLoader:
    def __init__(self, data_dir: str = None):
        """
        Initialize data loader
        Args:
            data_dir: Directory containing dataset
        """
        if data_dir is None:
            # 使用相对于项目根目录的路径
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        else:
            self.data_dir = data_dir
            
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.fusion = LateFusion(alpha=0.5)
        self.vector_store = VectorStore(dimension=768)

    def process_and_index(self, max_items: int = 1000):
        """
        Process dataset and index into vector store
        Args:
            max_items: Maximum number of items to process
        """
        dataset_path = os.path.join(self.data_dir, 'dataset.json')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Process items
        processed = 0
        for item in tqdm(dataset[:max_items], desc="Processing items"):
            try:
                # Load image
                image_path = os.path.join(self.data_dir, 'images', item['image_name'])
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                    
                image = Image.open(image_path)
                
                # Get embeddings
                image_embedding = self.image_encoder.encode(image)
                text_embedding = self.text_encoder.encode(item['description'])
                
                # Combine embeddings
                combined = self.fusion.combine(image_embedding, text_embedding)
                
                # Add to vector store
                metadata = {
                    'text': item['description'],
                    'image_url': image_path,  # 在实际部署时需要改为可访问的URL
                    'category': item.get('category', ''),
                    'attributes': item.get('attributes', {})
                }
                self.vector_store.add(combined.numpy(), metadata)
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing item {item.get('image_name', 'unknown')}: {e}")
                continue
            
        print(f"Successfully processed {processed} items")
        
        # 创建保存目录
        save_dir = os.path.join(self.data_dir, 'vector_store')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(save_dir)
        print(f"Vector store saved to {save_dir}")

def main():
    # 实例化并处理数据
    loader = DataLoader()
    loader.process_and_index(max_items=1000)

if __name__ == "__main__":
    main()