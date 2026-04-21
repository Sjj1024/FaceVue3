from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
from PIL import Image


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

    @staticmethod
    def _pil_to_rgb(image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
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
        """
        返回多个人脸 bbox: (x1, y1, x2, y2, score)
        Haar 不提供真实置信度，这里用 1.0 作为占位。
        """
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        cascade_paths = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
            Path(cv2.data.haarcascades) / "haarcascade_profileface.xml",
        ]

        # 多组参数从“严格→宽松”重试，提高命中率
        param_sets = [
            (1.1, 5, (60, 60)),
            (1.08, 4, (48, 48)),
            (1.05, 3, (32, 32)),
            (1.03, 3, (24, 24)),
        ]

        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []

        def add_box(x: int, y: int, w: int, h: int, score: float):
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # score：用面积近似（越大越可信），再乘一个常数占位
            area = max(1, int(w) * int(h))
            boxes.append((x1, y1, x2, y2))
            scores.append(float(score) * float(area))

        for cpath in cascade_paths:
            cascade = cv2.CascadeClassifier(str(cpath))
            if cascade.empty():
                continue

            for scale_factor, min_neighbors, min_size in param_sets:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=min_size,
                )
                if faces is not None and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        add_box(int(x), int(y), int(w), int(h), 1.0)

                # profileface 对“镜像侧脸”也试一次
                if "profileface" in cpath.name:
                    flipped = cv2.flip(gray, 1)
                    faces2 = cascade.detectMultiScale(
                        flipped,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        flags=cv2.CASCADE_SCALE_IMAGE,
                        minSize=min_size,
                    )
                    if faces2 is not None and len(faces2) > 0:
                        w_img = gray.shape[1]
                        for (x, y, w, h) in faces2:
                            # 翻转坐标映射回原图：x' = W - (x + w)
                            x_unflip = int(w_img - (int(x) + int(w)))
                            add_box(x_unflip, int(y), int(w), int(h), 1.0)

        if len(boxes) == 0:
            raise ValueError(
                "图片中未检测到人脸（Haar）。"
                "建议：保证脸更大更清晰、正脸、减少遮挡；"
                "如果是手机拍照请确保已做 EXIF 方向矫正（后端已处理）。"
            )

        # Haar 会对同一张脸产出多个相近框（不同 cascade/参数），
        # 这里把 NMS 阈值设置得更严格以减少重叠。
        keep = cls._nms(boxes, scores, iou_thresh=0.2, top_k=max_faces)
        out: list[tuple[int, int, int, int, float]] = []
        # 保持输出按面积从大到小，视觉上更稳定
        keep = sorted(
            keep,
            key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
            reverse=True,
        )
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            out.append((x1, y1, x2, y2, 1.0))
        return out

    @classmethod
    def _detect_face_bbox(cls, rgb: np.ndarray) -> tuple[int, int, int, int, float]:
        # 兼容旧逻辑：取最大的人脸
        faces = cls._detect_face_bboxes(rgb, max_faces=10)
        best = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return best

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
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < eps:
            return v
        return (v / n).astype(np.float32)

    # 对一张图里的一个人脸计算 embedding
    def embed_one(self, image: Image.Image) -> EmbedResult:
        self._ensure_session()
        rgb = self._pil_to_rgb(image)
        # 检测人脸
        x1, y1, x2, y2, score = self._detect_face_bbox(rgb)
        # 裁剪人脸
        face = rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        # 如果人脸裁剪失败，则抛出异常
        if face.size == 0:
            raise ValueError("人脸裁剪失败（bbox 越界）。")
        # 预处理人脸
        inp = self._preprocess_face(face)
        # 运行模型
        out = self._sess.run(None, {self._input_name: inp})  # type: ignore[arg-type]
        # 归一化 embedding
        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        # 对 embedding 进行 L2 归一化
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
        faces = self._detect_face_bboxes(rgb, max_faces=max_faces)
        results: list[EmbedResult] = []
        for (x1, y1, x2, y2, score) in faces:
            crop = rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue
            inp = self._preprocess_face(crop)
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

