## facevue3（人脸录入 / 识别 Demo）

一个前后端分离的本地/内网 Demo：

-   **录入（Enroll）**：上传/拍照 → 后端提取人脸特征（embedding）→ 写入 SQLite
-   **识别（Identify / 实时识别）**：上传/摄像头帧 → 提取 embedding → 与库中 embedding 做相似度检索 → 返回最像的人

---

## 技术原理（当前实现）

后端主要链路：

-   **人脸检测**：InsightFace（SCRFD）检测人脸框，并输出 **5 点关键点**
-   **对齐**：使用 5 点关键点将人脸对齐到 ArcFace 标准输入（112×112）
-   **特征提取**：ArcFace ONNX（onnxruntime）输出 **512 维 embedding**，并做 **L2 归一化**
-   **相似度**：余弦距离
    -   \(cos_sim = dot(a,b)\)
    -   \(cos_dist = 1 - cos_sim\)
-   **匹配判定**：当 `distance <= threshold` 判定为 `matched=true`，否则当作未匹配（未知）

数据存储：

-   SQLite：`backend/data/embeddings.db`

模型文件：

-   ArcFace ONNX：首次运行会自动下载到 `backend/models/arcface.onnx`
-   InsightFace 模型缓存：首次运行会下载到 `backend/models/insightface/`

---

## 目录结构

-   `src/`：Vue3 前端
-   `backend/`：FastAPI 后端
    -   `backend/main.py`：API 入口
    -   `backend/face_embedder.py`：检测 + 对齐 + embedding
    -   `backend/storage.py`：SQLite 存储

---

## 环境要求

-   Node.js（建议 18/20+）
-   pnpm
-   Python 3.10+（建议 3.11/3.12）

---

## 启动后端（FastAPI）

在项目根目录执行：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

打开接口文档：

-   `http://localhost:8000/docs`

### 首次启动提示

-   第一次调用 `/enroll` / `/identify` 之类接口时，会触发 InsightFace 模型下载与初始化，**可能需要几十秒到几分钟**（网络慢时更久）。
-   模型与缓存被写入项目目录 `backend/models/` 下，避免写入用户目录造成权限问题。

---

## 启动前端（Vue3 + Vite）

在项目根目录执行：

```bash
pnpm install
pnpm run dev
```

前端默认通过 Vite 代理走 `/api` 转发到后端（见 `src/pages/IdentifyPage.vue` 中的 `API_BASE = '/api'` 约定）。

---

## 使用流程（推荐）

-   **1) 先录入**：到“录入”页面给每个人多录几张（不同角度/光照/表情），样本越多越稳
-   **2) 再识别**：到“识别”页面上传图片或开启实时识别
-   **3) 调阈值**：
    -   `threshold` 越小越严格：更不容易误认，但“未知”会变多
    -   `threshold` 越大越宽松：更容易匹配，但误认风险更大

---

## FAQ（常见问题）

### 1) 前端请求一直 pending，后端看到 500（尤其是 `/enroll`）

常见原因是首次初始化 InsightFace 时写缓存目录无权限或下载/初始化很慢。

本项目已将 InsightFace 模型/缓存目录固定到 `backend/models/insightface/`，仍建议你：

-   **确保用 venv 启动后端**（不要用系统 Python）：

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

-   **首次初始化耐心等待**：第一次会下载模型并构建缓存，期间接口会慢

如果仍然 500，请在后端控制台查看 Python Traceback（完整堆栈）来定位网络/依赖问题。

### 2) 升级检测/对齐方案后，是否需要重新录入？

**建议重新录入（清空旧 embedding 再重录）**。因为裁剪/对齐策略变化会导致 embedding 分布变化，新旧混用会让距离判断不稳定。

清空方式（任选其一）：

-   删除数据库文件：`backend/data/embeddings.db`
-   或调用接口逐个删除：`DELETE /people/{person_id}`

### 3) `threshold` 是什么含义？

后端用余弦距离 `distance = 1 - dot(a,b)` 判断相似度。

-   `distance <= threshold` → 匹配（`matched=true`）
-   `distance > threshold` → 不匹配（`matched=false`）

---

## 隐私/合规提醒

人脸 embedding 属于生物识别信息。用于真实场景时，建议至少做到：

-   权限控制与审计
-   加密存储与备份策略
-   可删除与数据生命周期管理
-   明确告知与授权（合规要求以当地法规为准）
