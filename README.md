
# AI Video Search


## Tổng quan
AI Video Search là hệ thống tìm kiếm nội dung video đa phương tiện (text, audio, hình ảnh) tối ưu cho tiếng Việt, hỗ trợ:
- Phân cảnh tự động
- Nhận diện đối tượng, bối cảnh, text trong video
- Nhận diện và chuẩn hóa tiếng nói (ASR)
- Sinh vector embedding đa modal (hình, tiếng, text)
- Tìm kiếm nhanh, chính xác bằng truy vấn tự nhiên (text, voice, image)

## Cấu trúc thư mục
```
ai-video-search/
├── backend/        # Backend FastAPI, AI pipelines, DB connectors
├── frontend/       # Frontend React/Vite
├── shared/         # Thư viện, schema, utils dùng chung
├── docker/         # Dockerfile, docker-compose
```


## Vai trò các file chính

### backend/app/
- `main.py`: Entry point khởi động FastAPI, khai báo các route chính.
- `config.py`: Cấu hình đường dẫn model, DB, API key.
- `__init__.py`: Đánh dấu thư mục là package Python.

#### backend/app/api/
- `search.py`: Định nghĩa route API cho tìm kiếm video.
- `ingest.py`: Định nghĩa route API cho upload video, phân cảnh.
- `health.py`: Route kiểm tra tình trạng server.

#### backend/app/services/
- `scene_segmentation.py`: Phân cảnh video bằng TransNetV2.
- `visual_pipeline.py`: Xử lý keyframe, object detection, OCR, scene classification, visual embedding.
- `audio_pipeline.py`: Cắt audio, ASR, chuẩn hóa text, embedding audio & transcript.
- `fusion.py`: Gộp embedding đa modal, đồng bộ thời gian, ANN, fuzzy matching.
- `search_service.py`: Pipeline tìm kiếm, parse query, re-rank, RAG.

#### backend/app/models/
- `ocr_model.py`: Load và inference model OCR (VietOCR/PaddleOCR).
- `object_detector.py`: Load và inference model object detection (YOLO/Detectron2).
- `asr_model.py`: Load và inference model ASR (Whisper/Wav2Vec2).
- `embedding_model.py`: Load và encode embedding (CLIP, PhoBERT, BLIP2...)
- `fusion_model.py`: Model gộp embedding đa modal.

#### backend/app/db/
- `vector_db.py`: Kết nối và thao tác với FAISS/Milvus (vector DB).
- `metadata_db.py`: Kết nối và thao tác với Postgres/Elasticsearch (metadata DB).

#### backend/app/utils/
- `text_normalizer.py`: Chuẩn hóa tiếng Việt.
- `logger.py`: Tiện ích logging.
- `config_loader.py`: Đọc file config YAML.

#### backend/tests/
- `__init__.py`: Đánh dấu package test.

#### backend/requirements.txt
- Khai báo các package Python cần thiết cho backend.

### frontend/src/
#### api/
- `searchApi.ts`: Hàm gọi API tìm kiếm backend.
- `uploadApi.ts`: Hàm gọi API upload video backend.
#### components/
- `SearchBar.tsx`: Thanh tìm kiếm UI.
- `VideoPlayer.tsx`: Player video.
- `ScenePreview.tsx`: Hiển thị preview cảnh.
- `Filters.tsx`: Bộ lọc tìm kiếm.
#### pages/
- `Home.tsx`: Trang chủ.
- `SearchResults.tsx`: Trang kết quả tìm kiếm.
#### store/
- `index.ts`: Quản lý state (Redux/Zustand).
#### utils/
- `timeFormatter.ts`: Định dạng thời gian.
- `vnTextHighlight.ts`: Highlight từ khóa tiếng Việt.
#### styles/
- `index.css`: CSS/Tailwind.

### shared/
#### constants/
- `search_config.py`: Các hằng số cấu hình tìm kiếm.
#### schemas/
- `search_request.py`: Pydantic schema cho request tìm kiếm.
- `search_response.py`: Pydantic schema cho response tìm kiếm.
#### utils/
- `timecode.py`: Tiện ích chuyển đổi timecode.
- `vn_text.py`: Tiện ích xử lý tiếng Việt.

### docker/
- `backend.Dockerfile`: Dockerfile build backend.
- `frontend.Dockerfile`: Dockerfile build frontend.
- `docker-compose.yml`: Chạy đồng thời backend & frontend.


## Quy trình xử lý video (Pipeline tổng thể)
Hệ thống gồm 2 pipeline chính:

### 1. Offline Data Processing Pipeline (Indexing)
**Input:** Video (tiếng Việt hoặc đa ngôn ngữ, ưu tiên VN)

**Các bước:**
1. **Scene Segmentation**: Phân cảnh video bằng TransNetV2 → Xuất (scene_id, start_ts, end_ts)
2. **Visual Pipeline** (cho từng scene):
	- Keyframe extraction: lấy 3 frame (đầu, giữa, cuối)
	- Object detection: YOLOv8/Detectron2 → danh sách object + bbox
	- OCR: VietOCR/PaddleOCR (VN optimized) → text + vị trí
	- Scene classification: Indoor/Outdoor, context category
	- CLIP/BLIP2 embedding: vector đặc trưng hình ảnh
3. **Audio Pipeline** (cho từng scene):
	- Cắt audio theo scene
	- ASR: Whisper/Wav2Vec2 (VN) → transcript tiếng Việt
	- Text normalization: Chuẩn hóa chính tả, số, tên riêng
	- Audio embeddings: vector đặc trưng âm thanh
	- Transcript embeddings: từ transcript (PhoBERT/LaBSE)
4. **Multimodal Fusion**:
	- Temporal alignment: Đồng bộ frame và audio theo scene
	- Fusion embedding: gộp embedding visual (3 frame) + transcript + audio
	- Rapid matching (ANN/LSH): tăng tốc retrieval
	- Fuzzy matching: xử lý sai chính tả/âm tiếng Việt
	- Metadata: objects, OCR text, transcript, tags, scene labels
5. **Storage**:
	- Vector DB (FAISS/Milvus) → lưu fusion embedding
	- Metadata DB (Postgres + Elasticsearch) → lưu metadata để filter

**Ví dụ output cho 1 scene:**
```json
{
  "scene_id": "scene_12",
  "start_ts": "00:01:10.000",
  "end_ts": "00:01:15.000",
  "frames": [
	 {
		"frame_id": "scene_12_frame_1",
		"timestamp": "00:01:12.345",
		"objects": ["xe_may", "nguoi"],
		"ocr_text": "Ngân hàng ABC",
		"scene_label": "đường phố",
		"embedding_clip": [0.12, -0.05, ...]
	 },
	 // ...
  ],
  "audio": {
	 "transcript": "Ngân hàng ABC đang mở cửa.",
	 "embedding_text": [0.09, 0.14, ...],
	 "embedding_audio": [0.02, -0.03, ...]
  },
  "fusion_embedding": [0.11, -0.07, ...],
  "metadata": {
	 "objects": ["xe_may", "nguoi"],
	 "ocr_text": ["Ngân hàng ABC"],
	 "transcript": "Ngân hàng ABC đang mở cửa.",
	 "scene_label": "đường phố"
  }
}
```

### 2. Online Search Pipeline (Query → Result)
**Input:** Query tiếng Việt (text / voice / image)

**Các bước:**
1. (Optional) OpenAI LLM: Phân tích, mở rộng query → sinh bộ lọc metadata (ví dụ: "Ngân hàng ABC" trong phố Hà Nội năm 2020)
2. Nếu query là tiếng nói → ASR trước
3. **Metadata Filtering**: Elasticsearch query → lọc trước theo object, OCR text, transcript, thời gian, địa điểm
4. **Query Embedding**: Encode query (PhoBERT/CLIP text encoder) → vector
5. **ANN Search**: Truy vấn vector DB (fusion embeddings) → lấy top-K scene ứng viên
6. **Re-ranking**:
	- Cross-encoder: so sánh ngữ nghĩa sâu
	- Fuzzy text matching: xử lý sai chính tả tiếng Việt
	- Quality scoring: chọn scene rõ nét, âm thanh tốt
7. (Optional) OpenAI RAG: Lấy top-K → sinh tóm tắt, câu trả lời tiếng Việt
8. **Output:**
	- Scene preview (ảnh + audio snippet)
	- Timestamp trong video gốc
	- OCR text, transcript
	- Link mở video tại scene đó

---


## Công nghệ sử dụng

| Bước                  | Mô hình đề xuất tốt nhất                |
|-----------------------|-----------------------------------------|
| Scene Segmentation    | TransNetV2                              |
| Keyframe Extraction   | OpenCV + ffmpeg                         |
| Object Detection      | YOLOv8                                  |
| OCR                  | PaddleOCR                               |
| Scene Classification | Swin Transformer                        |
| Visual Embedding      | CLIP                                    |
| ASR                  | Whisper                                 |
| Text Normalization    | vncorenlp + custom rules                |
| Audio Embedding       | Wav2Vec2                                |
| Transcript Embedding  | PhoBERT (TV), LaBSE (đa ngôn ngữ)       |
| ANN Search            | FAISS                                   |
| Metadata DB           | PostgreSQL + Elasticsearch              |
| LLM, RAG              | OpenAI GPT (GPT-4 hoặc GPT-3.5)         |

📹 VIDEO INPUT
  ↓
🎬 SCENE SEGMENTATION (TransNetV2)
 - Phân cảnh dựa trên thay đổi khung hình lớn (scene boundaries)
 - Kết quả: danh sách (scene_id, start_ts, end_ts)

FOR EACH SCENE:

┌────────────────────────── VISUAL PIPELINE ────────────────────────────┐
│ 1) Keyframe extraction (first/mid/last frame)                         │
│ 2) Object detection (YOLOv8 / Detectron2) → danh sách đối tượng + bbox│
│ 3) OCR (PaddleOCR/VietOCR) → text tiếng Việt                          │
│ 4) Scene classification (ResNet/Swin)                                 │
│ 5) Visual embeddings (CLIP/BLIP2)                                     │
└───────────────────────────────────────────────────────────────────────┘

┌────────────────────────── AUDIO PIPELINE ─────────────────────────────┐
│ 1) Cắt audio theo start_ts, end_ts                                    │
│ 2) ASR tiếng Việt (Whisper large-v2 / Wav2Vec2-VN) → transcript       │
│ 3) Text normalization (chính tả, dấu, số)                             │
│ 4) Audio embeddings (Wav2Vec2/Hubert)                                 │
│ 5) Transcript embeddings (PhoBERT/LaBSE/embedding-VN)                 │
└───────────────────────────────────────────────────────────────────────┘

### MULTIMODAL FUSION

- Temporal alignment: đồng bộ frame + audio
- Rapid matching (ANN/LSH): tìm nearest neighbors nhanh
- Fuzzy matching (VN text): khớp gần đúng OCR/transcript
- Cross-modal fusion: gộp embedding audio + visual + text
- Output: fusion_embedding + metadata (objects, OCR text, transcript, tags, thời gian)

### STORAGE

- Vector DB (FAISS / Milvus): lưu fusion_embedding
- Metadata DB (Postgres + Elasticsearch): lưu metadata tìm kiếm được (cảnh, tag, đối tượng, text OCR, transcript)

### SEARCH PIPELINE

1. (Optional) OpenAI LLM: phân tích query tiếng Việt → filters (ngày, đối tượng, context)
2. Pre-filter metadata trong Elasticsearch
3. Encode query → embedding(s)
4. ANN search (fusion index + modality-specific index) → top-K ứng viên
5. Re-rank: cross-encoder + fuzzy text match + quality score
6. (Optional) OpenAI RAG: sinh câu trả lời / tóm tắt tiếng Việt
7. Trả về kết quả: preview cảnh, timestamp, trích đoạn transcript, OCR text
```


## Backend


## Frontend
- React, Vite, các component UI tìm kiếm, preview video
- Xem `frontend/package.json` để cài đặt


## Shared
- Schema Pydantic/TypeScript, utils, constants

docker-compose up --build

## Chạy nhanh bằng Docker
```bash
docker-compose up --build
```


## Phát triển local
### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
### Frontend
```bash
cd frontend
npm install
npm run dev
```
### Chạy Scene
python scene_segmentation.py --batch_folder ../../VIDEO_AIC2024_P1 --scenes scenes_all.json