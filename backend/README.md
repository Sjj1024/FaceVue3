# 本地后端：上传照片 → 计算人脸 embedding → 保存到 SQLite

这个后端用于**本地开发/内网部署**：上传一张照片（包含人脸），后端会用 **OpenCV Haar（找最大人脸）+ ArcFace ONNX（onnxruntime 推理）** 计算 embedding，并保存到本地 `SQLite`（`backend/data/embeddings.db`）。

## 目录结构

- `backend/main.py`: FastAPI 入口
- `backend/face_embedder.py`: embedding 计算（Haar 裁脸 + ArcFace ONNX）
- `backend/storage.py`: SQLite 存储（BLOB）
- `backend/data/embeddings.db`: 数据库（运行时自动创建）
- `backend/models/arcface.onnx`: ArcFace 模型（首次运行会自动下载）

## 环境要求

- Python 3.10+（建议 3.11/3.12；不建议用 3.14 这类太新的版本）

> 首次运行会自动下载模型到 `backend/models/arcface.onnx`。网络较慢时第一次会更久。

## 安装依赖

在项目根目录执行：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
uvicorn main:app --host 0.0.0.0 --port 8000
```

打开接口文档：

- `http://localhost:8000/docs`

## 上传并保存 embedding（enroll）

### curl 示例

```bash
curl -X POST "http://localhost:8000/enroll" \
  -F "name=张三" \
  -F "person_id=u_123" \
  -F "image=@/absolute/path/to/face.jpg"
```

返回示例：

```json
{
  "id": "9d7b0c8e-7c03-4f32-a1aa-0a8e2f2d2e65",
  "person_id": "u_123",
  "name": "张三",
  "model": "arcface-onnx:arcface.onnx",
  "dim": 512,
  "created_at": "2026-04-20T08:00:00Z"
}
```

> 说明：`person_id` 可不传，后端会自动生成 UUID；但 `name` 必传。

## 查询某个人已存的 embedding 条数

```bash
curl "http://localhost:8000/people/u_123"
```

## 查询已标注人员列表（people list）

返回已标注的人员（名字 + person_id），并带上该人已录入的 embedding 数量与最近一次录入时间。

```bash
curl "http://localhost:8000/people"
```

## 删除某个已标注人员（delete person）

删除该人的标注信息，并同时删除其所有 embeddings。

```bash
curl -X DELETE "http://localhost:8000/people/u_123"
```

## 识别是谁（identify）

上传一张照片，后端会计算 embedding 并在库里找“最像的 person_id”（余弦距离越小越像）。

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "top_k=3" \
  -F "threshold=0.35" \
  -F "image=@/absolute/path/to/query.jpg"
```

返回示例：

```json
{
  "matched": true,
  "threshold": 0.35,
  "best": { "person_id": "u_123", "name": "张三", "distance": 0.21, "embedding_id": "..." },
  "candidates": [
    { "person_id": "u_123", "name": "张三", "distance": 0.21, "embedding_id": "..." },
    { "person_id": "u_456", "name": "李四", "distance": 0.33, "embedding_id": "..." }
  ],
  "model": "arcface-onnx:arcface.onnx",
  "dim": 512
}
```

## 重要说明（隐私/合规）

- embedding 属于生物特征信息，建议至少做到：权限控制、加密存储、可删除、审计日志。

