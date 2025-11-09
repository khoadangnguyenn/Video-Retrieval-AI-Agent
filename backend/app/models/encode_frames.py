import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

# Add backend to path
sys.path.append("backend/app")

def load_ocr_data(video_id, data_root="data"):
    """Load OCR data từ file JSON"""
    ocr_path = Path(data_root) / "OCR" / f"{video_id}.json"
    if not ocr_path.exists():
        print(f"Không tìm thấy OCR data: {ocr_path}")
        return {}
    
    with open(ocr_path, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_frame_embeddings(video_id, max_frames=10, data_root="data"):
    """Encode N frames đầu tiên thành embeddings và lưu vào npz"""
    print(f"Bắt đầu encode embeddings cho video: {video_id}")
    
    # Import và khởi tạo model
    try:
        from models.embedding_model import EmbeddingModel
        print(" Import EmbeddingModel thành công")
        
        model = EmbeddingModel()
        print(" Khởi tạo model")
        
        model.load_model()  # Load tất cả models
        print(" Đã load models thành công")
    except Exception as e:
        print(f" Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load OCR data
    ocr_data = load_ocr_data(video_id, data_root)
    frames = ocr_data.get("frames", [])[:max_frames]
    
    if not frames:
        print(f"Không tìm thấy frames trong OCR data")
        return
    
    print(f"Sẽ xử lý {len(frames)} frames")
    
    # Tạo thư mục output
    embeddings_dir = Path(data_root) / "embeddings" / video_id
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Thư mục keyframes
    keyframes_dir = Path(data_root) / "keyframes" / video_id
    
    processed = 0
    for i, frame_info in enumerate(frames):
        frame_idx = frame_info.get("frame_idx", i+1)
        frame_name = frame_info.get("frame_name", f"{frame_idx:03d}.jpg")
        texts = frame_info.get("texts", [])
        
        print(f"Xử lý frame {frame_idx}: {frame_name}")
        
        # Đường dẫn ảnh
        image_path = keyframes_dir / frame_name
        if not image_path.exists():
            print(f"   Không tìm thấy ảnh: {image_path}")
            continue
        
        embeddings_dict = {}
        metadata = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "texts": texts
        }
        
        try:
            # Image embedding
            print(f"  - Encode image...")
            image_emb = model.embed_image(str(image_path))
            embeddings_dict["image"] = np.array(image_emb, dtype=np.float32)
            print(f"     Image embedding shape: {embeddings_dict['image'].shape}")
            
            # Text embedding (từ OCR)
            if texts:
                combined_text = " ".join(texts)
                print(f"  - Encode text: '{combined_text[:50]}...'")
                text_emb = model.embed_text(combined_text)
                embeddings_dict["text"] = np.array(text_emb, dtype=np.float32)
                print(f"     Text embedding shape: {embeddings_dict['text'].shape}")
            else:
                print(f"  - Không có text, skip text embedding")
                embeddings_dict["text"] = np.zeros(model.dimension, dtype=np.float32)
            
            # Lưu vào .npz file
            output_file = embeddings_dir / f"{frame_idx:03d}.npz"
            np.savez_compressed(
                output_file,
                **embeddings_dict,
                meta=json.dumps(metadata)
            )
            print(f"   Đã lưu: {output_file}")
            processed += 1
            
        except Exception as e:
            print(f"   Lỗi khi xử lý frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n Hoàn thành encode embeddings cho {processed}/{len(frames)} frames")
    print(f"Kết quả lưu tại: {embeddings_dir}")

if __name__ == "__main__":
    # Test với L21_V001, 10 frames đầu
    encode_frame_embeddings("L21_V001", max_frames=10, data_root="data")
