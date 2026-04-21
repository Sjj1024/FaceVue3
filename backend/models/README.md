# ArcFace ONNX 模型

默认情况下，后端会在首次计算 embedding 时自动下载模型到：

- `backend/models/arcface.onnx`

下载源（默认）：

- `https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true`

如果你在内网/离线环境部署，也可以手动把 ArcFace ONNX 文件放到上述路径。

