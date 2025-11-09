"""
Cross-encoder for ranking and reranking search results
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
from loguru import logger

from app.config import settings, MODEL_CONFIG

class CrossEncoder:
    """Cross-encoder model for reranking search results based on query-document relevance."""
    
    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        """
        Initialize cross-encoder model
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the cross-encoder model and tokenizer."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f" Cross-encoder model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f" Failed to load cross-encoder model: {e}")
            raise e
    
    def encode_pairs(self, query: str, documents: List[str]) -> List[float]:
        """
        Encode query-document pairs and compute relevance scores
        
        Args:
            query: Search query
            documents: List of document texts
            
        Returns:
            List of relevance scores for each document
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        scores = []
        
        for doc in documents:
            try:
                # Combine query and document
                pair_text = f"{query} [SEP] {doc}"
                
                # Tokenize
                inputs = self.tokenizer(
                    pair_text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token representation for scoring
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                    
                    # Simple scoring: compute similarity with a learned vector
                    # For now, use mean of the CLS embedding as score
                    score = torch.mean(cls_embedding).item()
                    scores.append(score)
                    
            except Exception as e:
                logger.warning(f"Failed to encode pair for document: {e}")
                scores.append(0.0)
        
        return scores
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank search candidates based on query relevance
        
        Args:
            query: Search query
            candidates: List of candidate documents with metadata
            top_k: Number of top results to return
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        # Extract text content from candidates
        documents = []
        for candidate in candidates:
            # Combine different text fields
            text_parts = []
            if 'transcript' in candidate:
                text_parts.append(candidate['transcript'])
            if 'ocr_text' in candidate:
                text_parts.append(candidate['ocr_text'])
            if 'caption' in candidate:
                text_parts.append(candidate['caption'])
            
            doc_text = " ".join(text_parts) if text_parts else ""
            documents.append(doc_text)
        
        # Compute relevance scores
        scores = self.encode_pairs(query, documents)
        
        # Add scores to candidates and sort
        for candidate, score in zip(candidates, scores):
            candidate['cross_encoder_score'] = score
        
        # Sort by cross-encoder score (descending)
        reranked = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        return reranked[:top_k]
    
    def batch_rerank(self, queries: List[str], candidates_list: List[List[Dict[str, Any]]], 
                    top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Batch reranking for multiple queries
        
        Args:
            queries: List of search queries
            candidates_list: List of candidate lists for each query
            top_k: Number of top results to return for each query
            
        Returns:
            List of reranked candidate lists
        """
        results = []
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates, top_k)
            results.append(reranked)
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test the cross-encoder
    cross_encoder = CrossEncoder()
    cross_encoder.load_model()
    
    # Sample data
    query = "tìm kiếm video về nấu ăn"
    candidates = [
        {
            "video_id": "video1",
            "transcript": "hôm nay tôi sẽ hướng dẫn các bạn nấu món phở",
            "ocr_text": "phở hà nội",
            "timestamp": "00:01:30"
        },
        {
            "video_id": "video2", 
            "transcript": "chúng ta cùng tìm hiểu về lịch sử Việt Nam",
            "ocr_text": "lịch sử",
            "timestamp": "00:02:15"
        }
    ]
    
    # Rerank
    reranked = cross_encoder.rerank(query, candidates, top_k=5)
    
    print("Reranked results:")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Video: {result['video_id']}, Score: {result['cross_encoder_score']:.4f}")
        print(f"   Content: {result.get('transcript', '')[:100]}...")
