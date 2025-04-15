from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """API configuration settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Multimodal Search API"
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    SAGEMAKER_ENDPOINT: str = "multimodal-search-endpoint"
    
    # Model Settings
    IMAGE_SIZE: int = 224
    MAX_TEXT_LENGTH: int = 512
    VECTOR_DIMENSION: int = 768
    
    # 本地模式标志（用于在SageMaker不可用时切换到本地模型）
    USE_LOCAL_MODEL: bool = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()