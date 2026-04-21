from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import uuid

from face_embedder import FaceEmbedder
from storage import EmbeddingStore


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DB_PATH = DATA_DIR / "embeddings.db"

app = FastAPI(title="Face Embedding Backend", version="0.1.0")

# 本地开发时给前端调用用（可按需收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EmbeddingStore(DB_PATH)
MODEL_PATH = APP_ROOT / "models" / "arcface.onnx"
embedder = FaceEmbedder(model_path=MODEL_PATH)


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/people")
def list_people():
    """
    查询已标注的人脸（people）列表。

    返回：person_id、name、该人 embedding 数量、最近一次录入时间等。
    """
    return {"items": [p.__dict__ for p in store.list_people()]}

@app.delete("/people/{person_id}")
def delete_person(person_id: str):
    """
    删除某个已标注的人（people）以及其所有 embeddings。
    """
    if not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id 不能为空。")
    res = store.delete_person(person_id.strip())
    if res["people_deleted"] == 0:
        raise HTTPException(status_code=404, detail="未找到该 person_id。")
    return res


@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    person_id: str | None = Form(None),
    image: UploadFile = File(...),
):
    """
    上传一张照片，计算 embedding，并保存到 SQLite。

    - person_id: 你系统里这个人的标识（例如 u_123）
    - image: 照片文件（jpg/png/webp...）
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="name 不能为空（例如：张三）。")
    pid = person_id.strip() if person_id and person_id.strip() else str(uuid.uuid4())
    # 读取图片内容
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="图片为空。")
    
    try:
        # 打开图片
        pil_img = Image.open(BytesIO(content))
        # 加载图片
        pil_img.load()
        # 转换图片方向
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")

    try:
        # 计算 embedding
        result = embedder.embed_one(pil_img)
        # 保存 embedding
        rec = store.enroll(
            person_id=pid,
            name=name.strip(),
            model=result.model,
            embedding=result.embedding,
            det_score=result.det_score,
            bbox_xyxy=result.bbox_xyxy,
        )
        return rec.__dict__
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding 计算失败：{e}")


@app.post("/identify")
async def identify(
    image: UploadFile = File(...),
    top_k: int = Form(3),
    threshold: float = Form(0.35),
):
    """
    上传一张照片，计算 embedding，并在库中识别“最像的 person_id”。

    - top_k: 返回候选数量
    - threshold: 余弦距离阈值（越小越像）。embedding 已做 L2 归一化：
      cos_sim = dot(a,b), cos_dist = 1 - cos_sim
    """
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k 必须在 1~20 之间。")
    if threshold <= 0 or threshold >= 2:
        raise HTTPException(status_code=400, detail="threshold 建议在 (0, 2) 之间。")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="图片为空。")

    try:
        pil_img = Image.open(BytesIO(content))
        pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")

    try:
        q = embedder.embed_one(pil_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding 计算失败：{e}")

    # 只在同一模型空间内比对
    db = store.list_all_embeddings(model=q.model)
    if len(db) == 0:
        return {
            "matched": False,
            "reason": "库中还没有任何 embedding（请先 enroll）。",
            "model": q.model,
            "dim": q.dim,
            "candidates": [],
        }

    # 计算余弦距离：1 - dot（因为都已 L2 归一化）
    qv = np.asarray(q.embedding, dtype=np.float32).reshape(-1)

    # 聚合到 person 级别：一个人可能有多条 embedding，取最小距离作为该人的分数
    best_by_person: dict[str, tuple[float, str]] = {}  # person_id -> (best_dist, embedding_id)
    for r in db:
        if r.dim != qv.size:
            continue
        dv = np.asarray(r.embedding, dtype=np.float32).reshape(-1)
        sim = float(np.dot(qv, dv))
        dist = float(1.0 - sim)
        cur = best_by_person.get(r.person_id)
        if cur is None or dist < cur[0]:
            best_by_person[r.person_id] = (dist, r.id)

    if not best_by_person:
        return {
            "matched": False,
            "reason": "库中 embedding 维度与查询不一致。",
            "model": q.model,
            "dim": q.dim,
            "candidates": [],
        }

    ranked = sorted(best_by_person.items(), key=lambda kv: kv[1][0])
    cand = []
    for person_id, (dist, emb_id) in ranked[:top_k]:
        cand.append({"person_id": person_id, "distance": dist, "embedding_id": emb_id})

    best = cand[0]
    matched = bool(best["distance"] <= threshold)

    return {
        "matched": matched,
        "threshold": threshold,
        "best": {**best, "name": store.get_person_name(best["person_id"])},
        "candidates": [
            {**c, "name": store.get_person_name(c["person_id"])} for c in cand
        ],
        "model": q.model,
        "dim": q.dim,
    }


@app.post("/identify_multi")
async def identify_multi(
    image: UploadFile = File(...),
    max_faces: int = Form(5),
    top_k: int = Form(3),
    threshold: float = Form(0.35),
):
    """
    多人脸识别：对一张图里的多张脸分别识别，并返回每张脸的 bbox + name 标注信息。
    """
    if max_faces < 1 or max_faces > 20:
        raise HTTPException(status_code=400, detail="max_faces 必须在 1~20 之间。")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k 必须在 1~20 之间。")
    if threshold <= 0 or threshold >= 2:
        raise HTTPException(status_code=400, detail="threshold 建议在 (0, 2) 之间。")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="图片为空。")

    try:
        pil_img = Image.open(BytesIO(content))
        pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")

    try:
        faces = embedder.embed_many(pil_img, max_faces=max_faces)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding 计算失败：{e}")

    # 用第一张脸的 model/dim 作为当前识别空间
    model = faces[0].model
    dim = faces[0].dim
    db = store.list_all_embeddings(model=model)
    if len(db) == 0:
        return {
            "model": model,
            "dim": dim,
            "threshold": threshold,
            "image": {"width": pil_img.size[0], "height": pil_img.size[1]},
            "faces": [],
            "reason": "库中还没有任何 embedding（请先 enroll）。",
        }

    def match_one(qv: np.ndarray):
        best_by_person: dict[str, tuple[float, str]] = {}
        for r in db:
            if r.dim != qv.size:
                continue
            dv = np.asarray(r.embedding, dtype=np.float32).reshape(-1)
            sim = float(np.dot(qv, dv))
            dist = float(1.0 - sim)
            cur = best_by_person.get(r.person_id)
            if cur is None or dist < cur[0]:
                best_by_person[r.person_id] = (dist, r.id)
        ranked = sorted(best_by_person.items(), key=lambda kv: kv[1][0])
        cand = []
        for person_id, (dist, emb_id) in ranked[:top_k]:
            cand.append(
                {
                    "person_id": person_id,
                    "name": store.get_person_name(person_id),
                    "distance": dist,
                    "embedding_id": emb_id,
                }
            )
        best = cand[0] if cand else None
        matched = bool(best and best["distance"] <= threshold)
        return matched, best, cand

    out_faces = []
    for f in faces:
        qv = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
        matched, best, cand = match_one(qv)
        x1, y1, x2, y2 = f.bbox_xyxy
        out_faces.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "det_score": f.det_score,
                "matched": matched,
                "best": best,
                "candidates": cand,
            }
        )

    return {
        "model": model,
        "dim": dim,
        "threshold": threshold,
        "image": {"width": pil_img.size[0], "height": pil_img.size[1]},
        "faces": out_faces,
    }


@app.get("/people/{person_id}")
def count_person(person_id: str):
    return {"person_id": person_id, "count": store.count_person(person_id)}

