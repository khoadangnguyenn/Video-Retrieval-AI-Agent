import whisper
import logging

logger = logging.getLogger(__name__)

class ASRModel:
    def __init__(self, model_path="base"):
        try:
            if "whisper-" in model_path:
                model_path = model_path.split("whisper-")[-1]
            logger.info(f"Loading Whisper model: {model_path}")
            self.model = whisper.load_model(model_path)
            logger.info(f"Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise e
    
    def transcribe(self, audio_path):
        try:
            result = self.model.transcribe(audio_path)
            transcript = result["text"].strip()
            logger.debug(f"Whisper transcription: {transcript[:50]}...")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
