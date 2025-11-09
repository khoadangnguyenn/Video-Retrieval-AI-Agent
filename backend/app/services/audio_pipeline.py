# ASR PIPELINE using Whisper - Process ALL keyframes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import subprocess
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.models.asr_model import ASRModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WhisperASRWrapper:
    def __init__(self, model_path="base"):
        try:
            logger.info(f"Loading Whisper ASR model: {model_path}")
            self.asr_model = ASRModel(model_path=model_path)
            logger.info("Whisper ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise e
            
    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return ""
        try:
            return self.asr_model.transcribe(audio_path)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

class AudioPipeline:
    def __init__(self, asr_model=None, data_dir="data"):
        self.asr_model = asr_model
        self.data_dir = Path(data_dir)
        self.audio_segments_dir = self.data_dir / "Audio" / "whisper_segments"
        self.asr_output_dir = self.data_dir / "ASR"
        self.map_keyframes_dir = self.data_dir / "map-keyframes"
        
        self.audio_segments_dir.mkdir(parents=True, exist_ok=True)
        self.asr_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AudioPipeline initialized")

    def _find_ffmpeg(self) -> Optional[str]:
        possible_paths = [
            r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            "ffmpeg"
        ]
        for path in possible_paths:
            try:
                result = subprocess.run([path, "-version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"Found ffmpeg at: {path}")
                    return path
            except:
                continue
        return None

    def create_keyframe_csv_from_images(self, video_id: str) -> bool:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        keyframes_dir = os.path.join(project_root, "data", "keyframes", video_id)
        csv_path = self.map_keyframes_dir / f"{video_id}.csv"
        
        if not os.path.exists(keyframes_dir):
            return False
            
        images = sorted([f for f in os.listdir(keyframes_dir) if f.endswith(".jpg")])
        if not images:
            return False
        
        csv_content = "n,pts_time,fps,frame_idx\n"
        for i, img in enumerate(images):
            pts_time = i * 2.5
            frame_idx = i * 75
            csv_content += f"{i+1},{pts_time},30.0,{frame_idx}\n"
        
        self.map_keyframes_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        
        logger.info(f"Created {csv_path} with {len(images)} keyframes")
        return True

    def process_video_with_keyframes(self, video_id: str, max_segments: int = None) -> Dict[str, Any]:
        logger.info(f"=== PROCESSING {video_id} ===")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        video_path = os.path.join(project_root, "data", "videos", f"{video_id}.mp4")
        csv_path = self.map_keyframes_dir / f"{video_id}.csv"
        
        if not os.path.exists(video_path):
            error_msg = f"Video not found: {video_path}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        if not csv_path.exists():
            if not self.create_keyframe_csv_from_images(video_id):
                error_msg = f"Could not create keyframe CSV for {video_id}"
                logger.error(error_msg)
                return {"error": error_msg}
        
        try:
            df = pd.read_csv(csv_path)
            total_keyframes = len(df)
            logger.info(f"Loaded {total_keyframes} keyframes from CSV")
            
            if max_segments is None:
                max_segments = total_keyframes - 1
                logger.info(f"Processing ALL {max_segments} segments")
            else:
                logger.info(f"Processing up to {max_segments} segments")
                
        except Exception as e:
            error_msg = f"Failed to load CSV: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        output_dir = self.audio_segments_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            error_msg = "FFmpeg not found!"
            logger.error(error_msg)
            return {"error": error_msg}
        
        successful_cuts = 0
        successful_transcripts = 0
        results = []
        actual_segments = min(max_segments, len(df) - 1)
        
        logger.info(f"Starting processing of {actual_segments} audio segments...")
        
        for i in range(actual_segments):
            start_time = df.iloc[i]["pts_time"]
            end_time = df.iloc[i + 1]["pts_time"] - 0.1
            duration = max(1.0, end_time - start_time)
            
            output_filename = f"{video_id}_seg_{i+1:03d}.wav"
            output_path = output_dir / output_filename
            
            cmd = [
                ffmpeg_path, "-i", video_path,
                "-ss", str(start_time), "-t", str(duration),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", str(output_path)
            ]
            
            logger.info(f"[{i+1:03d}/{actual_segments}] Cutting {output_filename}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0 and output_path.exists():
                    size_kb = output_path.stat().st_size // 1024
                    successful_cuts += 1
                    
                    transcript = ""
                    if self.asr_model:
                        try:
                            transcript = self.asr_model.transcribe(str(output_path))
                            if transcript:
                                successful_transcripts += 1
                        except Exception as e:
                            logger.warning(f"Transcription failed: {e}")
                    
                    segment_data = {
                        "segment_id": f"seg_{i+1:03d}",
                        "file": output_filename,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "size_kb": size_kb,
                        "transcript": transcript
                    }
                    results.append(segment_data)
                    
                    if transcript:
                        display_text = transcript[:40] + "..." if len(transcript) > 40 else transcript
                        logger.info(f"    Success ({size_kb}KB) - \"{display_text}\"")
                    else:
                        logger.info(f"    Success ({size_kb}KB) - No transcript")
                else:
                    logger.error(f"    Failed to cut {output_filename}")
            except Exception as e:
                logger.error(f"    Error: {e}")
        
        success_rate = (successful_cuts / actual_segments * 100) if actual_segments > 0 else 0
        transcript_rate = (successful_transcripts / successful_cuts * 100) if successful_cuts > 0 else 0
        
        result = {
            "video_id": video_id,
            "total_keyframes": total_keyframes,
            "total_segments": actual_segments,
            "successful_cuts": successful_cuts,
            "successful_transcriptions": successful_transcripts,
            "success_rate": success_rate,
            "transcription_rate": transcript_rate,
            "segments": results
        }

        # Save ASR results to file in data/ASR
        try:
            asr_output_path = self.asr_output_dir / f"{video_id}.json"
            with open(asr_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"ASR results saved to {asr_output_path}")
        except Exception as e:
            logger.error(f"Failed to save ASR results: {e}")

        logger.info(f"=== {video_id} SUMMARY ===")
        logger.info(f"Audio segments: {successful_cuts}/{actual_segments} ({success_rate:.1f}%)")
        logger.info(f"Transcriptions: {successful_transcripts}/{successful_cuts} ({transcript_rate:.1f}%)")

        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper ASR Audio Pipeline")
    parser.add_argument("--asr-model", default="base", help="Whisper model: tiny, base, small, medium, large")
    parser.add_argument("--max-segments", type=int, default=None, help="Maximum segments to process")
    parser.add_argument("--all", action="store_true", help="Process ALL keyframes")
    args = parser.parse_args()
    
    if args.all:
        args.max_segments = None
    
    logger.info("=== WHISPER ASR PIPELINE ===")
    
    try:
        asr_model = WhisperASRWrapper(model_path=args.asr_model)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        data_dir = os.path.join(project_root, "data")
        
        audio_pipeline = AudioPipeline(asr_model=asr_model, data_dir=data_dir)
        
        test_videos = ["L21_V001", "L21_V002"]
        overall_results = {}
        total_success = 0
        
        for video_id in test_videos:
            result = audio_pipeline.process_video_with_keyframes(
                video_id=video_id,
                max_segments=args.max_segments
            )
            
            overall_results[video_id] = result
            
            if "error" in result:
                logger.error(f"{video_id} FAILED: {result['error']}")
            else:
                logger.info(f"{video_id} SUCCESS")
                total_success += 1
            
            print()
        
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Successfully processed: {total_success}/{len(test_videos)} videos")
        
        if total_success > 0:
            total_segments = sum(r.get("successful_cuts", 0) for r in overall_results.values() if "error" not in r)
            total_transcripts = sum(r.get("successful_transcriptions", 0) for r in overall_results.values() if "error" not in r)
            
            logger.info(f"Total audio segments created: {total_segments}")
            logger.info(f"Total Whisper transcripts: {total_transcripts}")
            logger.info("Audio pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
