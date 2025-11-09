"""
OCR model loader (PaddleOCR only) - Complete version with text-only output
Fixed for PaddleOCR 3.2+ API compatibility
"""
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Union, Optional


class OCRModel:
    def __init__(self, model_path: str = None, lang: str = "vi", gpu: bool = True):
        """
        Initialize OCR model (PaddleOCR only) - Updated for 3.2+ API
        """
        self.model_path = model_path
        self.lang = lang
        self.gpu = gpu
        self.model = None
        self.model_type = "paddle"

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install with: pip install paddleocr paddlepaddle"
            )

        # Init PaddleOCR with simplified parameters for 3.2+ API
        init_errors = []
        
        # Try different initialization methods - Updated for PaddleOCR 3.2+
        try_methods = [
            lambda: PaddleOCR(lang=self.lang),
            lambda: PaddleOCR()
        ]

        for i, method in enumerate(try_methods):
            try:
                self.model = method()
                print(f"[INFO] Initialized PaddleOCR model (method {i+1}, lang={self.lang})")
                break
            except Exception as e:
                init_errors.append(f"Method {i+1}: {str(e)}")
                continue

        if self.model is None:
            raise RuntimeError(f"Failed to initialize PaddleOCR. Errors: {'; '.join(init_errors)}")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _extract_text_paddle(self, image_path: str) -> List[Dict[str, Union[str, List, float]]]:
        """Extract text using PaddleOCR 3.2+ API with robust parsing"""
        try:
            # Updated API call for PaddleOCR 3.2+
            results = self.model.predict(image_path)
        except Exception as e:
            print(f"[ERROR] PaddleOCR failed: {e}")
            return []

        extracted_texts = []
        if not results or results is None:
            print("[WARN] No text detected")
            return []

        threshold = 0.5

        # Handle PaddleOCR 3.2+ output format
        try:
            # New format: results contain OCRResult objects with .json attribute
            for result in results:
                if hasattr(result, 'json') and 'res' in result.json:
                    res_data = result.json['res']
                    texts = res_data.get('rec_texts', [])
                    scores = res_data.get('rec_scores', [])
                    boxes = res_data.get('dt_polys', [])
                    
                    for j, (text, score) in enumerate(zip(texts, scores)):
                        if isinstance(text, str) and text.strip() and float(score) > threshold:
                            box = boxes[j] if j < len(boxes) else []
                            extracted_texts.append({
                                "text": text.strip(),
                                "bbox": box,
                                "confidence": float(score)
                            })

            # Fallback for older format compatibility  
            if not extracted_texts and isinstance(results, list) and len(results) > 0:
                if results[0] is not None:
                    items = results[0] if isinstance(results[0], list) else results
                    
                    for entry_idx, line in enumerate(items):
                        try:
                            if not line or len(line) != 2:
                                continue
                                
                            box, text_info = line
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text, score = text_info[0], text_info[1]
                            elif isinstance(text_info, str):
                                text, score = text_info, 1.0
                            else:
                                continue

                            if isinstance(text, str) and text.strip() and float(score) > threshold:
                                extracted_texts.append({
                                    "text": text.strip(),
                                    "bbox": box,
                                    "confidence": float(score)
                                })
                                
                        except Exception as e:
                            print(f"[WARN] Error processing entry {entry_idx}: {e}")
                            continue

        except Exception as e:
            print(f"[ERROR] Error parsing OCR results: {e}")
            return []

        print(f"[INFO] Found {len(extracted_texts)} valid text regions")
        return extracted_texts

    def extract_text(self, image_path: str, data_root: str = "data") -> List[Dict[str, Union[str, List, float]]]:
        """Extract text from image"""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return []

        if self.model is None:
            print("[ERROR] OCR model not initialized")
            return []

        try:
            results = self._extract_text_paddle(image_path)
            # Save JSON with text-only format
            self._save_image_ocr_json(image_path, results, data_root)
            return results
        except Exception as e:
            print(f"[ERROR] OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _save_image_ocr_json(self, image_path: str, results: List[Dict], data_root: str = "data"):
        """Save OCR results for a single image (text-only format)"""
        try:
            image_path_obj = Path(image_path)
            ocr_dir = Path(data_root) / "OCR"
            ocr_dir.mkdir(parents=True, exist_ok=True)

            # Extract only text
            texts = []
            for result in results:
                if isinstance(result.get("text"), str) and result["text"].strip():
                    texts.append(result["text"].strip())

            # Save simplified output (text-only)
            out = {
                "image": str(image_path_obj.as_posix()),
                "frame_name": image_path_obj.name,
                "texts": texts
            }

            out_path = ocr_dir / f"{image_path_obj.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            print(f"[INFO] OCR JSON saved to {out_path}")
            return out_path
        except Exception as e:
            print(f"[ERROR] Could not write OCR JSON: {e}")
            return None

    def process_video_keyframes(self, video_id: str, data_root: str = "data") -> Dict:
        """Process all keyframes of a video and save OCR results (text-only format)"""
        keyframe_dir = Path(data_root) / "keyframes" / video_id
        ocr_dir = Path(data_root) / "OCR"
        ocr_dir.mkdir(parents=True, exist_ok=True)

        if not keyframe_dir.exists():
            print(f"[ERROR] Keyframe directory not found: {keyframe_dir}")
            return {"error": "Keyframe directory not found"}

        results = []
        frames = sorted([f for f in keyframe_dir.glob("*.jpg")], key=lambda x: int(x.stem))
        total_frames = len(frames)

        print(f"[INFO] Processing {total_frames} keyframes for video {video_id}")

        for i, frame_path in enumerate(frames, 1):
            frame_idx = int(frame_path.stem)
            print(f"\n[{i}/{total_frames}] Processing frame {frame_idx}")

            ocr_results = self._extract_text_paddle(str(frame_path))
            
            # Collect only text (simplified format)
            frame_texts = []
            for result in ocr_results:
                if isinstance(result.get("text"), str) and result["text"].strip():
                    frame_texts.append(result["text"].strip())
            
            if frame_texts:
                results.append({
                    "frame_idx": frame_idx,
                    "frame_name": frame_path.name,
                    "texts": frame_texts
                })
            
            print(f"Found {len(frame_texts)} text regions in frame {frame_idx}")

        # Save results với tên file giống như tên folder trong keyframes
        folder_name = keyframe_dir.name  # Lấy tên folder thực tế
        output = {
            "video_id": folder_name,  # Sử dụng tên folder thực tế
            "total_frames_processed": total_frames,
            "frames_with_text": len(results),
            "frames": results
        }

        # Lưu file JSON theo tên folder thực tế
        output_path = ocr_dir / f"{folder_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] OCR results saved to {output_path}")
        print(f"[INFO] Summary: {len(results)}/{total_frames} frames contain text")
        return output


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OCR Text Extraction Tool - PaddleOCR 3.2+ Compatible')
    parser.add_argument('--image', help='Path to test single image')
    parser.add_argument('--video-id', help='Process all keyframes for a video')
    parser.add_argument('--lang', default='vi', help='Language code (default: vi)')
    parser.add_argument('--data-root', default='data', help='Root directory for data (default: data)')

    args = parser.parse_args()

    try:
        print(f"[INFO] Initializing OCR model with PaddleOCR 3.2+ API...")
        ocr = OCRModel(lang=args.lang)
        print(f"[INFO] OCR model initialized successfully")

        if args.video_id:
            print(f"\n[INFO] Processing video keyframes: {args.video_id}")
            results = ocr.process_video_keyframes(args.video_id, args.data_root)
            if "error" not in results:
                print(f"\n=== VIDEO OCR SUMMARY ===")
                print(f"Video ID: {args.video_id}")
                print(f"Total frames with text: {len(results['frames'])}")
                    
        elif args.image:
            print(f"\n[INFO] Processing single image: {args.image}")
            results = ocr.extract_text(args.image, args.data_root)
            texts = [r["text"] for r in results if r.get("text")]
            
            print(f"\n=== IMAGE OCR RESULTS ===")
            print(f"Image: {args.image}")
            print(f"Total text regions found: {len(texts)}")
            
            for i, text in enumerate(texts, 1):
                confidence = next((r["confidence"] for r in results if r["text"] == text), 0)
                print(f"{i:2d}. {text} (confidence: {confidence:.3f})")
                
        else:
            parser.print_help()
            print(f"\nExamples:")
            print(f"  Single image: python {__file__} --image path/to/image.jpg --lang vi")
            print(f"  Video frames: python {__file__} --video-id L21_V001 --lang vi")

    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        import traceback
        traceback.print_exc()
