## facevue3 (Face Enroll / Identify Demo)

[English](README.md) | [中文](README_ZH.md)

A simple local / intranet demo with a separated frontend and backend:

-   **Enroll**: upload/capture a photo → backend extracts face embedding → stored in SQLite
-   **Identify (incl. real-time)**: upload/camera frame → extract embedding → similarity search in DB → return the closest person

---

## How it works (current implementation)

Backend pipeline:

-   **Face detection**: InsightFace (SCRFD) detects face bbox and returns **5-point landmarks**
-   **Alignment**: align the face to ArcFace canonical input (112×112) using the 5 landmarks
-   **Embedding**: ArcFace ONNX (onnxruntime) outputs a **512-d embedding** and we apply **L2 normalization**
-   **Similarity**: cosine distance
    -   \(cos_sim = dot(a,b)\)
    -   \(cos_dist = 1 - cos_sim\)
-   **Match decision**: `matched=true` when `distance <= threshold`, otherwise treated as unknown/unmatched

Storage:

-   SQLite: `backend/data/embeddings.db`

Models:

-   ArcFace ONNX: downloaded on first run to `backend/models/arcface.onnx`
-   InsightFace model cache: downloaded on first run to `backend/models/insightface/`

---

## Project structure

-   `src/`: Vue 3 frontend
-   `backend/`: FastAPI backend
    -   `backend/main.py`: API entry
    -   `backend/face_embedder.py`: detection + alignment + embedding
    -   `backend/storage.py`: SQLite storage

---

## Requirements

-   Node.js (recommended 18/20+)
-   pnpm
-   Python 3.10+ (recommended 3.11/3.12)

---

## Start the backend (FastAPI)

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs:

-   `http://localhost:8000/docs`

### First-run notes

-   The first call to `/enroll` / `/identify` may trigger InsightFace model download + initialization and can take **tens of seconds to minutes** (depending on network).
-   Model/cache files are written under `backend/models/` to avoid permission issues in user home directories.

---

## Start the frontend (Vue 3 + Vite)

From the repository root:

```bash
pnpm install
pnpm run dev
```

The frontend uses Vite proxy and calls the backend via `/api` (see `API_BASE = '/api'` in `src/pages/IdentifyPage.vue`).

---

## Recommended workflow

-   **1) Enroll first**: enroll multiple samples per person (different angles/lighting/expressions) for better stability
-   **2) Then identify**: upload a photo or enable real-time identify
-   **3) Tune threshold**:
    -   smaller `threshold` = stricter = fewer false positives, more unknowns
    -   larger `threshold` = looser = fewer unknowns, higher false-positive risk

---

## FAQ

### 1) The frontend request stays pending and the backend returns 500 (especially `/enroll`)

Most commonly, this happens during InsightFace first-time initialization (model download / cache write permissions / slow initialization).

This project pins InsightFace model/cache root to `backend/models/insightface/`. Still make sure you:

-   **Start the backend with the venv** (do not use system Python):

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

-   **Wait for first-run initialization**: the first request may be slow while models are downloaded and caches are built

If you still get 500, check the Python traceback in the backend logs to locate the exact cause (network/permission/dependency).

### 2) After upgrading detection/alignment, do I need to re-enroll?

**Yes, it is recommended to re-enroll (clear old embeddings and enroll again)**. Changing crop/alignment changes the embedding distribution; mixing old and new embeddings makes distance thresholds unstable.

Ways to clear:

-   Delete the database file: `backend/data/embeddings.db`
-   Or delete people one by one: `DELETE /people/{person_id}`

### 3) What does `threshold` mean?

The backend uses cosine distance `distance = 1 - dot(a,b)`.

-   `distance <= threshold` → matched (`matched=true`)
-   `distance > threshold` → not matched (`matched=false`)

---

## Privacy / compliance note

Face embeddings are biometric data. For real use cases, you should implement at least:

-   access control and audit logs
-   encryption at rest and backup strategy
-   deletion / lifecycle management
-   user notice and consent (follow local regulations)
