from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL import ImageOps


@dataclass(frozen=True)
class EmbedResult:
    embedding: np.ndarray  # float32, shape (dim,)
    dim: int
    model: str
    bbox_xyxy: tuple[float, float, float, float]
    det_score: float


DEFAULT_MODEL_URL = (
    "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true"
)


class FaceEmbedder:
    """
    本地可跑的“照片 → embedding”：

    1) OpenCV Haar 找最大人脸（返回 bbox）
    2) ArcFace ONNX（onnxruntime）输出 512 维 embedding

    说明：
    - Haar 只是为了减少依赖、保证易安装；精度比 SCRFD/RetinaFace 差一些，但足够做 demo/内网使用。
    - embedding 会做 L2 归一化（便于余弦相似度检索）。
    """

    def __init__(self, model_path: str | Path, model_url: str = DEFAULT_MODEL_URL):
        self.model_path = Path(model_path)
        self.model_url = model_url
        self._sess: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[list[int | str | None]] = None
        self._fa = None  # InsightFace FaceAnalysis（检测 + 5点关键点）
        # InsightFace 默认写 ~/.insightface，在受限环境可能无权限；改到项目内可写目录
        self._insight_root = Path(__file__).resolve().parent / "models" / "insightface"

    def _ensure_model(self):
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(self.model_url, self.model_path)

    def _ensure_session(self):
        if self._sess is not None:
            return
        self._ensure_model()
        sess = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._sess = sess
        inp0 = sess.get_inputs()[0]
        self._input_name = inp0.name
        # shape 可能是 [1, 3, 112, 112] 或 [1, 112, 112, 3]，也可能带 None/字符串
        self._input_shape = list(inp0.shape)

    def _ensure_face_analyzer(self):
        """
        使用 InsightFace 的 SCRFD 检测（输出 bbox + 5 点关键点），用于更准确的裁剪/对齐。
        """
        if self._fa is not None:
            return
        # InsightFace / Matplotlib / fontconfig 会写用户目录缓存；在受限环境可能无权限，
        # 统一改到项目内可写目录，避免初始化卡住或报 500。
        os.environ.setdefault("INSIGHTFACE_HOME", str(self._insight_root))
        os.environ.setdefault("MPLCONFIGDIR", str(self._insight_root / ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self._insight_root / ".cache"))
        os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 insightface，无法启用更准确的人脸检测/对齐。"
                "请先 pip install -r backend/requirements.txt"
            ) from e

        self._insight_root.mkdir(parents=True, exist_ok=True)
        fa = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=str(self._insight_root),
        )
        # det_size 越大越准但更慢；这里取 640 兼顾准确性与速度
        fa.prepare(ctx_id=0, det_size=(640, 640))
        self._fa = fa

    @staticmethod
    def _pil_to_rgb(image: Image.Image) -> np.ndarray:
        # 统一处理 EXIF 方向，避免旋转导致检测/裁剪偏差
        img = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
        return rgb

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = a_area + b_area - inter
        return float(inter / union) if union > 0 else 0.0

    @classmethod
    def _nms(
        cls,
        boxes: list[tuple[int, int, int, int]],
        scores: list[float],
        iou_thresh: float = 0.2,
        top_k: int = 10,
    ) -> list[int]:
        idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        keep: list[int] = []
        for i in idxs:
            if len(keep) >= top_k:
                break
            ok = True
            for j in keep:
                if cls._iou(boxes[i], boxes[j]) >= iou_thresh:
                    ok = False
                    break
            if ok:
                keep.append(i)
        return keep

    @classmethod
    def _detect_face_bboxes(
        cls, rgb: np.ndarray, *, max_faces: int = 5
    ) -> list[tuple[int, int, int, int, float]]:
        raise NotImplementedError("请使用实例方法 _detect_faces（需要 FaceAnalysis）。")

    def _detect_faces(self, rgb: np.ndarray, *, max_faces: int = 5):
        """
        返回 InsightFace 检测到的人脸对象列表（包含 bbox、det_score、kps）。
        """
        self._ensure_face_analyzer()
        faces = self._fa.get(rgb)  # type: ignore[union-attr]
        if not faces:
            raise ValueError(
                "图片中未检测到人脸（SCRFD）。建议：保证脸更大更清晰、正脸、减少遮挡。"
            )
        # InsightFace 内部已做 NMS；这里按面积从大到小取前 max_faces，稳定且符合原 UI 预期
        faces = sorted(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            reverse=True,
        )[: max_faces]
        return faces

    @classmethod
    def _detect_face_bbox(cls, rgb: np.ndarray) -> tuple[int, int, int, int, float]:
        raise NotImplementedError("请使用实例方法 embed_one（需要 kps 对齐）。")

    def _preprocess_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """
        ArcFace ONNX 常用预处理：
        - resize 到 112x112
        - (img - 127.5) / 128.0
        - 输出布局依模型输入决定（NCHW 或 NHWC）
        """
        import cv2

        resized = cv2.resize(rgb_face, (112, 112), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32)
        x = (x - 127.5) / 128.0

        # 默认按 NCHW；若模型输入是 NHWC 则不转置
        shape = self._input_shape or []
        # 仅判断常见 4 维输入
        if len(shape) == 4:
            # NHWC: [N, 112, 112, 3]
            if (shape[1] == 112 and shape[2] == 112 and shape[3] == 3) or (
                shape[3] == 3 and shape[1] in (112, None, "None") and shape[2] in (112, None, "None")
            ):
                x = np.expand_dims(x, axis=0)  # NHWC
                return x

            # NCHW: [N, 3, 112, 112]
            if (shape[1] == 3 and shape[2] == 112 and shape[3] == 112) or (
                shape[1] == 3 and shape[2] in (112, None, "None") and shape[3] in (112, None, "None")
            ):
                x = np.transpose(x, (2, 0, 1))  # CHW
                x = np.expand_dims(x, axis=0)  # NCHW
                return x

        # 兜底：先尝试 NCHW（更常见）
        x_nchw = np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0)
        return x_nchw

    @staticmethod
    def _aligned_crop(rgb: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        使用 5 点关键点把人脸对齐到 ArcFace 标准 112x112。
        """
        from insightface.utils import face_align

        kps5 = np.asarray(kps, dtype=np.float32)
        if kps5.shape != (5, 2):
            raise ValueError("关键点格式不正确，无法对齐。")
        aligned_bgr = face_align.norm_crop(rgb[..., ::-1], kps5)  # InsightFace 期望 BGR
        aligned_rgb = aligned_bgr[..., ::-1].copy()
        return aligned_rgb

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < eps:
            return v
        return (v / n).astype(np.float32)

    # 对一张图里的一个人脸计算 embedding
    def embed_one(self, image: Image.Image) -> EmbedResult:
        self._ensure_session()
        # 转换图片为 RGB
        rgb = self._pil_to_rgb(image)
        # 检测 + 5点对齐：比 Haar 裁剪稳定很多
        f0 = self._detect_faces(rgb, max_faces=1)[0]
        x1, y1, x2, y2 = [int(v) for v in f0.bbox.tolist()]
        score = float(getattr(f0, "det_score", 1.0))
        kps = getattr(f0, "kps", None)
        if kps is None:
            raise ValueError("检测器未返回关键点，无法进行对齐。")
        aligned = self._aligned_crop(rgb, kps)
        inp = self._preprocess_face(aligned)
        # 运行模型
        out = self._sess.run(None, {self._input_name: inp})  # type: ignore[arg-type]
        # 归一化 embedding
        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        # 对 embedding 进行 L2 归一化？L2 归一化是什么？
        # L2 归一化是将向量除以其 L2 范数，使得向量模长为 1，从而避免不同向量之间模长的影响
        emb = self._l2_normalize(emb)
        # 返回 embedding 结果
        return EmbedResult(
            embedding=emb,
            dim=int(emb.shape[0]),
            model=f"arcface-onnx:{self.model_path.name}",
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            det_score=float(score),
        )

    def embed_many(self, image: Image.Image, *, max_faces: int = 5) -> list[EmbedResult]:
        """
        对一张图里的多个人脸计算 embedding（最多 max_faces 张）。
        """
        self._ensure_session()
        rgb = self._pil_to_rgb(image)
        faces = self._detect_faces(rgb, max_faces=max_faces)
        results: list[EmbedResult] = []
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox.tolist()]
            score = float(getattr(f, "det_score", 1.0))
            kps = getattr(f, "kps", None)
            if kps is None:
                continue
            aligned = self._aligned_crop(rgb, kps)
            inp = self._preprocess_face(aligned)
            out = self._sess.run(None, {self._input_name: inp})  # type: ignore[arg-type]
            emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
            emb = self._l2_normalize(emb)
            results.append(
                EmbedResult(
                    embedding=emb,
                    dim=int(emb.shape[0]),
                    model=f"arcface-onnx:{self.model_path.name}",
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    det_score=float(score),
                )
            )
        if len(results) == 0:
            raise ValueError("检测到人脸但未能生成 embedding（裁剪失败或推理失败）。")
        return results

