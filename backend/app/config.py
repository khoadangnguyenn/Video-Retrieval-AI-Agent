"""
Configuration settings for AI Video Search Backend
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Video Search"
    VERSION: str = "1.0.0"
    
    # Database Configuration
    POSTGRES_URL: str = "postgresql://user:password@localhost/ai_video_search"
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = "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"
    
    # Vector Database Configuration
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    VECTOR_DIMENSION: int = 768  # PhoBERT embedding dimension
    
    # Model Paths
    MODELS_DIR: str = "./models"
    PHOBERT_MODEL_PATH: str = "vinai/phobert-base"
    CLIP_MODEL_PATH: str = "openai/clip-vit-base-patch32"
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    WHISPER_MODEL_PATH: str = "base"  # or "small", "medium", "large"
    
    # Processing Configuration
    MAX_VIDEO_SIZE_MB: int = 500
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".avi", ".mov", ".mkv"]
    SUPPORTED_AUDIO_FORMATS: list = [".mp3", ".wav", ".m4a"]
    SUPPORTED_IMAGE_FORMATS: list = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # Scene Segmentation
    SCENE_SEGMENTATION_THRESHOLD: float = 0.3
    MIN_SCENE_DURATION: float = 1.0  # seconds
    
    # Keyframe Extraction
    KEYFRAMES_PER_SCENE: int = 3
    
    # Object Detection
    OBJECT_DETECTION_CONFIDENCE: float = 0.5
    MAX_OBJECTS_PER_FRAME: int = 20
    
    # OCR Configuration
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    
    # ASR Configuration
    ASR_LANGUAGE: str = "vi"  # Vietnamese
    ASR_TASK: str = "transcribe"
    
    # Search Configuration
    SEARCH_TOP_K: int = 20
    RE_RANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Vietnamese Text Processing
    VIETNAMESE_NORMALIZATION: bool = True
    REMOVE_TONES: bool = False
    
    # Storage Configuration
    UPLOAD_DIR: str = "./uploads"
    PROCESSED_DIR: str = "./processed"
    CACHE_DIR: str = "./cache"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Additional settings that might be in environment
    DEBUG: bool = False
    SECRET_KEY: str = "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"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore extra fields from environment
    }

# Global settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.CACHE_DIR,
        settings.MODELS_DIR,
        os.path.dirname(settings.LOG_FILE),
        os.path.dirname(settings.FAISS_INDEX_PATH)
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "phobert": {
        "model_name": settings.PHOBERT_MODEL_PATH,
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "clip": {
        "model_name": settings.CLIP_MODEL_PATH,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "yolo": {
        "model_path": settings.YOLO_MODEL_PATH,
        "conf": settings.OBJECT_DETECTION_CONFIDENCE,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "whisper": {
        "model_name": settings.WHISPER_MODEL_PATH,
        "language": settings.ASR_LANGUAGE,
        "task": settings.ASR_TASK,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

# Vietnamese text processing configuration
VIETNAMESE_CONFIG = {
    "normalization": settings.VIETNAMESE_NORMALIZATION,
    "remove_tones": settings.REMOVE_TONES,
    "lowercase": True,
    "remove_punctuation": False
}

# Fusion weights configuration
FUSION_WEIGHTS = {
    "text": 0.45,      # Transcripts, OCR, captions
    "image": 0.35,     # Frames, images
    "audio": 0.20      # Speech-to-text embeddings or audio embeddings
}

# Initialize directories on import
ensure_directories()
