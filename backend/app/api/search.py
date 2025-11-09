"""
Search API endpoints for the AI Video Search system
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import time
from loguru import logger

from app.services.search_service import search_service
from app.utils.text_normalizer import text_normalizer
from app.config import settings
from app.services.llm_service import LLMService

router = APIRouter()


@router.post("/text")
async def search_by_text(
    query: str = Form(..., description="Vietnamese text query"),
    top_k: int = Query(20, description="Number of results to return"),
    use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
    filters: Optional[str] = Form(None, description="JSON string of additional filters")
):
    """
    Search videos by Vietnamese text query
    """
    try:
        start_time = time.time()
        
        # Parse filters if provided
        parsed_filters = None
        if filters:
            import json
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
        # Normalize query
        normalized_query = text_normalizer.normalize_text(query)
        
        logger.info(f"Text search request: {query} -> {normalized_query}")
        
        # Perform search
        results = await search_service.search(
            query=normalized_query,
            query_type="text",
            filters=parsed_filters,
            top_k=top_k,
            use_rag=use_rag
        )
        
        # Add search time and query
        search_time = time.time() - start_time
        results["search_time"] = search_time
        results["query"] = query
        
        logger.info(f"Text search completed in {search_time:.2f}s, found {results['total_results']} results")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Text search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/image")
async def search_by_image(
    image: UploadFile = File(..., description="Image file for similarity search"),
    query: Optional[str] = Form(None, description="Additional text query"),
    top_k: int = Query(20, description="Number of results to return"),
    use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
    filters: Optional[str] = Form(None, description="JSON string of additional filters")
):
    """
    Search videos by image similarity
    """
    try:
        start_time = time.time()
        
        # Validate image file
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Parse filters if provided
        parsed_filters = None
        if filters:
            import json
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
        logger.info(f"Image search request: {image.filename}, size: {len(image_bytes)} bytes")
        
        # Perform search
        results = await search_service.search(
            query=query or "",
            query_type="image",
            query_image=image_bytes,
            filters=parsed_filters,
            top_k=top_k,
            use_rag=use_rag
        )
        
        # Add search time and query
        search_time = time.time() - start_time
        results["search_time"] = search_time
        results["query"] = query or f"Image search: {image.filename}"
        
        logger.info(f"Image search completed in {search_time:.2f}s, found {results['total_results']} results")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Image search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/audio")
async def search_by_audio(
    audio: UploadFile = File(..., description="Audio file for speech search"),
    query: Optional[str] = Form(None, description="Additional text query"),
    top_k: int = Query(20, description="Number of results to return"),
    use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
    filters: Optional[str] = Form(None, description="JSON string of additional filters")
):
    """
    Search videos by audio/speech content
    """
    try:
        start_time = time.time()
        
        # Validate audio file
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio bytes
        audio_bytes = await audio.read()
        
        # Parse filters if provided
        parsed_filters = None
        if filters:
            import json
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
        logger.info(f"Audio search request: {audio.filename}, size: {len(audio_bytes)} bytes")
        
        # Perform search
        results = await search_service.search(
            query=query or "",
            query_type="audio",
            query_audio=audio_bytes,
            filters=parsed_filters,
            top_k=top_k,
            use_rag=use_rag
        )
        
        # Add search time and query
        search_time = time.time() - start_time
        results["search_time"] = search_time
        results["query"] = query or f"Audio search: {audio.filename}"
        
        logger.info(f"Audio search completed in {search_time:.2f}s, found {results['total_results']} results")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Audio search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/suggest")
async def get_search_suggestions(
    query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(10, description="Number of suggestions to return")
):
    """
    Get search suggestions based on partial query
    """
    try:
        # Normalize query
        normalized_query = text_normalizer.normalize_text(query)
        
        # Extract keywords
        keywords = text_normalizer.extract_keywords(normalized_query)
        
        # Generate suggestions (simple approach)
        suggestions = []
        
        # Add keyword-based suggestions
        for keyword in keywords[:limit//2]:
            suggestions.append(f"Tìm kiếm {keyword}")
            suggestions.append(f"Video có {keyword}")
        
        # Add common Vietnamese search patterns
        common_patterns = [
            "người",
            "xe",
            "nhà",
            "đường",
            "cửa hàng",
            "công viên",
            "trường học",
            "bệnh viện"
        ]
        
        for pattern in common_patterns:
            if pattern not in suggestions and len(suggestions) < limit:
                suggestions.append(pattern)
        
        return JSONResponse(content={
            "query": query,
            "suggestions": suggestions[:limit]
        })
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")


@router.get("/filters")
async def get_available_filters():
    """
    Get available filters for search
    """
    try:
        filters = {
            "objects": [
                "người", "xe", "cây", "nhà", "đường", "cửa hàng",
                "công viên", "trường học", "bệnh viện", "xe máy",
                "ô tô", "xe đạp", "mèo", "chó", "chim"
            ],
            "locations": [
                "hà nội", "sài gòn", "hồ chí minh", "tp.hcm", "hcm",
                "đà nẵng", "huế", "nha trang", "vũng tàu"
            ],
            "time_of_day": [
                "sáng", "trưa", "chiều", "tối", "đêm"
            ],
            "weather": [
                "nắng", "mưa", "mây", "trời quang"
            ]
        }
        
        return JSONResponse(content=filters)
        
    except Exception as e:
        logger.error(f"Get filters failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get filters: {str(e)}")


@router.get("/statistics")
async def get_search_statistics():
    """
    Get search system statistics
    """
    try:
        # Get vector database statistics
        vector_stats = search_service.vector_db.get_statistics()
        
        # Get metadata database statistics
        metadata_stats = await search_service.metadata_db.get_statistics()
        
        # Get embedding model info
        embedding_info = {
            "phobert_loaded": "phobert" in search_service.embedding_model.models,
            "clip_loaded": "clip" in search_service.embedding_model.models,
            "asr_loaded": search_service.asr_model.model is not None,
            "cross_encoder_loaded": search_service.cross_encoder.model is not None
        }
        
        # Get LLM service info
        llm_info = {
            "openai_available": LLMService.is_available(),
            "offline_mode": not settings.USE_LLM
        }
        
        statistics = {
            "vector_database": vector_stats,
            "metadata_database": metadata_stats,
            "models": embedding_info,
            "llm_service": llm_info,
            "system_info": {
                "version": settings.VERSION,
                "vector_dimension": settings.VECTOR_DIMENSION,
                "fusion_enabled": True
            }
        }
        
        return JSONResponse(content=statistics)
        
    except Exception as e:
        logger.error(f"Get statistics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/health")
async def health_check():
    """
    Health check for search service
    """
    try:
        # Check if all components are available
        health_status = {
            "status": "healthy",
            "components": {
                "embedding_models": "phobert" in search_service.embedding_model.models,
                "asr_model": search_service.asr_model.model is not None,
                "cross_encoder": search_service.cross_encoder.model is not None,
                "vector_db": search_service.vector_db.indices["text"] is not None,
                "metadata_db": search_service.metadata_db.es_client is not None,
                "llm_service": LLMService.is_available(),
                "fusion_service": True  # Always available
            }
        }
        
        # Check if all components are healthy
        all_healthy = all(health_status["components"].values())
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(content={
            "status": "unhealthy",
            "error": str(e)
        }, status_code=500)