"""
OpenAI LLM service for query parsing and RAG functionality
"""
import openai
from typing import Dict, Any, List, Optional
from loguru import logger

from app.config import settings


class LLMService:
    """OpenAI LLM service for Vietnamese query processing and RAG"""
    
    def __init__(self):
        self.client = None
        self.api_key = None
        self.use_llm = settings.USE_LLM
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.use_llm:
            logger.info("LLM features disabled - using offline mode")
            return
            
        try:
            # Get API key from environment or config
            self.api_key = settings.OPENAI_API_KEY
            
            if self.api_key:
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found, LLM features will be disabled")
                self.use_llm = False
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
            self.use_llm = False
    
    async def parse_and_expand_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse and expand Vietnamese query using OpenAI or rule-based fallback
        
        Args:
            query: Original Vietnamese query
            filters: Existing filters
            
        Returns:
            Parsed query with extracted filters and expanded terms
        """
        if not self.use_llm or not self.client:
            logger.info("Using rule-based query parsing (offline mode)")
            return self._rule_based_query_parsing(query, filters)
        
        try:
            system_prompt = """
            Bạn là một trợ lý AI chuyên xử lý truy vấn tìm kiếm video tiếng Việt. 
            Nhiệm vụ của bạn là phân tích truy vấn và trích xuất thông tin có cấu trúc.
            
            Hãy phân tích truy vấn và trả về JSON với các trường sau:
            - original_query: truy vấn gốc
            - normalized_query: truy vấn đã chuẩn hóa
            - keywords: danh sách từ khóa quan trọng
            - query_type: loại truy vấn (visual/audio/text)
            - extracted_filters: các bộ lọc được trích xuất
            - expanded_terms: các từ khóa mở rộng liên quan
            - confidence: độ tin cậy của việc phân tích (0-1)
            
            Ví dụ truy vấn: "Tìm video có người đi bộ trên đường phố Hà Nội vào buổi sáng"
            """
            
            user_prompt = f"""
            Truy vấn: "{query}"
            
            Hãy phân tích và trả về kết quả dưới dạng JSON.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed_result = self._parse_llm_response(content)
            
            # Merge with existing filters
            if filters:
                parsed_result["extracted_filters"].update(filters)
            
            logger.info(f"LLM parsed query: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM query parsing failed: {str(e)}")
            return self._rule_based_query_parsing(query, filters)
    
    def _rule_based_query_parsing(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback rule-based query parsing"""
        from app.utils.text_normalizer import text_normalizer
        
        normalized_query = text_normalizer.normalize_text(query)
        keywords = text_normalizer.extract_keywords(query)
        
        # Simple rule-based filter extraction
        extracted_filters = {}
        query_lower = query.lower()
        
        # Location extraction
        locations = ["hà nội", "sài gòn", "hồ chí minh", "tp.hcm", "hcm", "đà nẵng", "huế"]
        for location in locations:
            if location in query_lower:
                extracted_filters["location"] = location
                break
        
        # Time extraction
        time_patterns = {"sáng": "morning", "trưa": "noon", "chiều": "afternoon", "tối": "evening", "đêm": "night"}
        for vn_time, en_time in time_patterns.items():
            if vn_time in query_lower:
                extracted_filters["time_of_day"] = en_time
                break
        
        # Object extraction
        objects = ["người", "xe", "cây", "nhà", "đường", "cửa hàng", "công viên"]
        found_objects = [obj for obj in objects if obj in query_lower]
        if found_objects:
            extracted_filters["objects"] = found_objects
        
        # Query type detection
        if any(word in query_lower for word in ["hình", "ảnh", "video", "cảnh"]):
            query_type = "visual"
        elif any(word in query_lower for word in ["tiếng", "âm thanh", "nói", "hát"]):
            query_type = "audio"
        else:
            query_type = "text"
        
        result = {
            "original_query": query,
            "normalized_query": normalized_query,
            "keywords": keywords,
            "query_type": query_type,
            "extracted_filters": extracted_filters,
            "expanded_terms": keywords,  # Simple expansion
            "confidence": 0.7
        }
        
        # Merge with existing filters
        if filters:
            result["extracted_filters"].update(filters)
        
        return result
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON"""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            # Return default structure
            return {
                "original_query": "",
                "normalized_query": "",
                "keywords": [],
                "query_type": "text",
                "extracted_filters": {},
                "expanded_terms": [],
                "confidence": 0.0
            }
    
    async def generate_rag_answer(self, scenes: List[Dict[str, Any]], query: str) -> Optional[str]:
        """
        Generate RAG answer using OpenAI or return None if disabled
        
        Args:
            scenes: List of relevant scenes
            query: Original query
            
        Returns:
            Generated answer in Vietnamese or None if LLM disabled
        """
        if not self.use_llm or not self.client or not scenes:
            logger.info("RAG disabled - returning None")
            return None
        
        try:
            # Prepare context from scenes
            context = self._prepare_rag_context(scenes)
            
            system_prompt = """
            Bạn là một trợ lý AI chuyên tìm kiếm video. Dựa trên thông tin từ các cảnh video, 
            hãy trả lời câu hỏi của người dùng một cách ngắn gọn và chính xác bằng tiếng Việt.
            """
            
            user_prompt = f"""
            Câu hỏi: "{query}"
            
            Thông tin từ các cảnh video:
            {context}
            
            Hãy trả lời câu hỏi dựa trên thông tin trên.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated RAG answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"RAG answer generation failed: {str(e)}")
            return None
    
    def _prepare_rag_context(self, scenes: List[Dict[str, Any]]) -> str:
        """Prepare context for RAG from scenes"""
        context_parts = []
        
        for i, scene in enumerate(scenes[:5], 1):  # Limit to top 5 scenes
            scene_info = f"Cảnh {i}: "
            
            if scene.get("transcript"):
                scene_info += f"Transcript: {scene['transcript']} "
            if scene.get("ocr_text"):
                scene_info += f"OCR: {scene['ocr_text']} "
            if scene.get("detected_objects"):
                scene_info += f"Objects: {', '.join(scene['detected_objects'])} "
            if scene.get("scene_description"):
                scene_info += f"Mô tả: {scene['scene_description']} "
            
            context_parts.append(scene_info)
        
        return "\n".join(context_parts)
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.use_llm and self.client is not None


# Global LLM service instance
llm_service = LLMService()