import faiss
import numpy as np
from typing import List, Tuple, Dict
import json
import os

class VectorStore:
    def __init__(self, dimension: int = 768):
        """
        Initialize FAISS vector store.
        Args:
            dimension: Dimension of the vectors to be stored
        """
        # Initialize FAISS index
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2距离度量
        
        # 存储元数据
        self.metadata: Dict[int, Dict] = {}
        self.next_id = 0
        
    def add(self, vector: np.ndarray, metadata: Dict) -> int:
        """
        Add a vector and its metadata to the store.
        Args:
            vector: Embedding vector
            metadata: Associated metadata (e.g., text, image_path)
        Returns:
            int: Index of the added vector
        """
        # 确保向量格式正确
        vector = vector.reshape(1, -1).astype(np.float32)
        
        # 添加到FAISS索引
        self.index.add(vector)
        
        # 存储元数据
        self.metadata[self.next_id] = metadata
        self.next_id += 1
        
        return self.next_id - 1
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar vectors.
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
        Returns:
            List of (id, distance, metadata) tuples
        """
        # 确保向量格式正确
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 搜索最近邻
        distances, indices = self.index.search(query_vector, k)
        
        # 组合结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS返回-1表示无效结果
                results.append((int(idx), float(dist), self.metadata[int(idx)]))
                
        return results
        
    def save(self, directory: str):
        """
        Save the index and metadata to disk.
        Args:
            directory: Directory to save the files
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # 保存元数据
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump({
                "metadata": self.metadata,
                "next_id": self.next_id,
                "dimension": self.dimension
            }, f)
            
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load the index and metadata from disk.
        Args:
            directory: Directory containing the saved files
        Returns:
            VectorStore instance
        """
        # 加载元数据
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            data = json.load(f)
            
        # 创建实例
        store = cls(dimension=data["dimension"])
        store.metadata = {int(k): v for k, v in data["metadata"].items()}
        store.next_id = data["next_id"]
        
        # 加载FAISS索引
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        return store