from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import time
import torch
import numpy as np
from PIL import Image
import io

from ..models import SearchQuery, SearchResponse, SearchResult
from ...models.image_encoder import ImageEncoder
from ...models.text_encoder import TextEncoder
from ...models.fusion import LateFusion
from ...utils.vector_store import VectorStore

router = APIRouter()

# 初始化模型和向量存储
image_encoder = ImageEncoder()
text_encoder = TextEncoder()
fusion = LateFusion(alpha=0.5)
vector_store = VectorStore(dimension=768)  # BERT/ViT维度

@router.post("/search", response_model=SearchResponse)
async def search(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(10)
):
    """
    搜索端点，支持多模态输入
    """
    start_time = time.time()
    
    # 获取embeddings
    image_embedding = None
    text_embedding = None
    
    if image:
        contents = await image.read()
        image_tensor = image_encoder.encode_file(contents)
        image_embedding = image_tensor.numpy()
        
    if text:
        text_tensor = text_encoder.encode(text)
        text_embedding = text_tensor.numpy()
    
    # 融合特征
    if image_embedding is not None or text_embedding is not None:
        combined = fusion.combine(
            torch.from_numpy(image_embedding) if image_embedding is not None else None,
            torch.from_numpy(text_embedding) if text_embedding is not None else None
        ).numpy()
        
        # 搜索向量库
        results = vector_store.search(combined, k=top_k)
        
        # 格式化结果
        search_results = [
            SearchResult(
                id=str(idx),
                score=float(1.0 / (1.0 + dist)),  # 转换距离为相似度分数
                text=metadata.get("text", ""),
                image_url=metadata.get("image_url")
            )
            for idx, dist, metadata in results
        ]
    else:
        search_results = []
    
    query_time = time.time() - start_time
    
    return SearchResponse(
        results=search_results,
        query_time=query_time
    )

@router.post("/index")
async def add_to_index(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    添加新项目到索引
    """
    # 获取embeddings
    image_embedding = None
    text_embedding = text_encoder.encode(text).numpy()
    
    if image:
        contents = await image.read()
        image_embedding = image_encoder.encode_file(contents).numpy()
    
    # 融合特征
    combined = fusion.combine(
        torch.from_numpy(image_embedding) if image_embedding is not None else None,
        torch.from_numpy(text_embedding)
    ).numpy()
    
    # 添加到向量存储
    metadata = {
        "text": text,
        "image_url": image_url
    }
    idx = vector_store.add(combined, metadata)
    
    return {"id": str(idx)}