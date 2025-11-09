"""
Embedding model loader
    PhoBERT for text
    CLIP for image
"""

import torch
import io
import os
import soundfile as sf
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

class EmbeddingModel:

    def get_asr_transcript(self, video_id: str, segment_id: str, asr_dir="data/ASR") -> str:
        """Lấy transcript từ file ASR json cho video_id và segment_id."""
        import json
        from pathlib import Path
        asr_path = Path(asr_dir) / f"{video_id}.json"
        if not asr_path.exists():
            return ""
        try:
            with open(asr_path, "r", encoding="utf-8") as f:
                asr_data = json.load(f)
            for seg in asr_data.get("segments", []):
                if seg.get("segment_id") == segment_id:
                    return seg.get("transcript", "")
        except Exception:
            pass
        return ""


    async def embed_audio(self, video_id: str, segment_id: str, mode: str = "audio", data_root="data") -> list:
        """
        Nhúng audio segment:
        - mode="text": lấy transcript từ ASR json và nhúng text (PhoBERT)
        - mode="audio": nhúng trực tiếp file wav (Wav2Vec2)
        """
        from pathlib import Path
        audio_path = Path(data_root) / "Audio" / "whisper_segments" / video_id / f"{video_id}_{segment_id}.wav"
        if mode == "text":
            transcript = self.get_asr_transcript(video_id, segment_id)
            if transcript:
                return await self.embed_text(transcript)
            else:
                return [0.0] * self.dimension
        elif mode == "audio":
            # Nhúng trực tiếp audio bằng Wav2Vec2
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return [0.0] * self.dimension
            import soundfile as sf
            audio_input, sampling_rate = sf.read(str(audio_path), dtype='float32')
            inputs = self.wav2vec2_processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate)
            with torch.no_grad():
                embeddings = self.wav2vec2_model(**inputs).last_hidden_state.mean(dim=1)
            return embeddings.tolist()
        else:
            raise ValueError("mode must be 'text' or 'audio'")
    def __init__(self):
        # CLIP for image
        self.clip_processor = None
        self.clip_model = None

        # PhoBERT for text
        self.phobert_tokenizer = None
        self.phobert_model = None

        # Wav2Vec2 for audio
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

        self.dimension = 768  # Common embedding dimension for text and audio

    def load_model(self):
        """Load all embedding models."""
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # PhoBERT for text
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Wav2Vec2 for audio
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        print("Embedding models loaded.")

    async def embed_image(self, image_data: bytes) -> list:
        """Encode image to embedding vector using CLIP."""
        inputs = self.clip_processor(images=image_data, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        return embeddings.tolist()

    async def embed_text(self, text_data: str) -> list:
        """Encode text to embedding vector using PhoBERT."""
        inputs = self.phobert_tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.phobert_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.tolist()

