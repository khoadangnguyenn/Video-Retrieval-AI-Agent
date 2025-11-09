# Data Directory Structure

Thư mục `data/` chứa tất cả dữ liệu đầu vào, trung gian và kết quả của hệ thống AI Video Search.

##  Cấu trúc thư mục

###  **videos/**
- **Mục đích**: Lưu trữ video gốc
- **Format**: `.mp4`, `.avi`, `.mov`
- **Quy tắc đặt tên**: `{video_id}.mp4`
- **Ví dụ**: `L21_V001.mp4`, `L21_V002.mp4`

###  **keyframes/**
- **Mục đích**: Lưu trữ keyframes được trích xuất từ video
- **Cấu trúc**: `keyframes/{video_id}/*.jpg`
- **Ví dụ**: `keyframes/L21_V001/001.jpg`
- **Tạo bởi**: Video processing pipeline

###  **map-keyframes/**
- **Mục đích**: Lưu mapping thời gian của keyframes
- **Format**: CSV với cột `n,pts_time,fps,frame_idx`
- **Ví dụ**: `L21_V001.csv`, `L21_V002.csv`
- **Sử dụng bởi**: Audio pipeline để tạo segments

###  **Audio/**
- **Mục đích**: Lưu audio segments được cắt từ video
- **Cấu trúc**: 
  - `Audio/transformers_segments/{video_id}/*.wav` - Whisper/Transformers ASR
  - `Audio/whisper_segments/{video_id}/*.wav` - Direct Whisper
- **Tạo bởi**: `audio_pipeline.py`

###  **ASR/** (Automatic Speech Recognition)
- **Mục đích**: Lưu kết quả phiên âm từ audio
- **Formats**:
  - `{video_id}_transcripts.json` - Định dạng JSON
  - `{video_id}_transcripts.csv` - Định dạng CSV  
  - `{video_id}_transformers/` - File TXT riêng biệt cho từng segment
- **Tạo bởi**: Audio pipeline với ASR models

###  **OCR/** (Optical Character Recognition)
- **Mục đích**: Lưu kết quả trích xuất text từ keyframes
- **Formats**:
  - `{video_id}_ocr_results.json` - Định dạng JSON
  - `{video_id}_ocr_results.csv` - Định dạng CSV
  - `{video_id}_ocr/` - File TXT riêng biệt cho từng keyframe
- **Tạo bởi**: OCR processing pipeline

###  **embeddings/**
- **Mục đích**: Lưu vector embeddings cho multimodal search
- **Files**:
  - `{video_id}_image_embeddings.npy` - Image feature vectors
  - `{video_id}_text_embeddings.npy` - Text feature vectors
  - `{video_id}_audio_embeddings.npy` - Audio feature vectors
  - `{video_id}_fused_embeddings.npy` - Multimodal fused vectors
- **Tạo bởi**: Embedding models (CLIP, Sentence-BERT)

###  **search-results/**
- **Mục đích**: Lưu kết quả tìm kiếm và ranking
- **Files**:
  - `query_{timestamp}.json` - Kết quả search với scores
  - `rankings/{query_id}_rankings.csv` - Kết quả đã được rank
  - `cache/` - Cache kết quả để tăng performance
- **Tạo bởi**: Search và ranking systems

###  **AIC25 Dataset Directories**

#### **clip-features-32-aic25-b1/**
- **Mục đích**: CLIP features từ AIC25 Batch 1 dataset
- **Files**: `{video_id}.npy`

#### **media-info-aic25-b1/media-info/**
- **Mục đích**: Metadata về video files
- **Format**: JSON với thông tin duration, resolution, etc.

#### **objects-aic25-b1/objects/**
- **Mục đích**: Object detection và annotations
- **Format**: JSON với bounding boxes, object classes, timestamps

##  Workflow

1. **Input**: Videos vào `videos/`
2. **Extract**: Keyframes  `keyframes/`
3. **Map**: Timing mapping  `map-keyframes/`
4. **Process**: 
   - Audio segments  `Audio/`
   - ASR transcripts  `ASR/`
   - OCR text  `OCR/`
5. **Embed**: Feature vectors  `embeddings/`
6. **Search**: Query results  `search-results/`

##  Notes

- Tất cả `.gitkeep` files giúp duy trì cấu trúc thư mục trong Git
- Các thư mục được tạo tự động bởi pipeline khi cần thiết
- Backup dữ liệu quan trọng thường xuyên
- Xóa dữ liệu tạm thời định kỳ để tiết kiệm dung lượng
