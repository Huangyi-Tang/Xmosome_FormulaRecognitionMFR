# Xmosome_FormulaRecognitionMFR

[English](README.md) | [简体中文](README_CN.md)

将 [Pix2Text-MFR 1.5](https://huggingface.co/breezedeus/pix2text-mfr-1.5) 离线转换为 Apple Core ML（FP16）模型包，供 Xmosome 使用。

应用侧用途：

- macOS — 屏幕截图中的公式识别
- iPadOS / iOS — Apple Pencil 手写公式转 LaTeX

推理全程在设备本地完成（Neural Engine / GPU）。本仓库只负责构建与打包；客户端在运行时加载生成的模型包。

## 技术路线

| 项目 | 说明 |
| --- | --- |
| 上游模型 | [breezedeus/pix2text-mfr-1.5](https://huggingface.co/breezedeus/pix2text-mfr-1.5) — TrOCR 风格 DeiT-small 编码器 + 因果解码器，词表 1868，MIT |
| 推理运行时 | Core ML 8.x `mlprogram`，FP16 |
| 序列分桶 | 解码长度 `[32, 64, 128, 256]`，避免动态形状反复编译 |
| 上游发布格式 | Hugging Face 以 ONNX 权重发布；打包时转为 Core ML。App 运行时不使用 ONNX |

## 模型包结构

`./scripts/build-pack.sh` 输出目录规范如下（同时会打成 zip）：

```text
FormulaRecognitionMFR/
├── pack.json             # 清单（format 版本 1）
├── MFREncoder.mlmodelc/  # 视觉编码器  [1, 3, 384, 384] → [1, 578, 384]
├── MFRDecoder.mlmodelc/  # 序列解码器 [1, L] + 编码特征 → [1, L, 1868]
├── vocab.json            # Byte-level BPE 词表与分桶配置
└── LICENSE-Pix2Text.txt  # 上游 MIT 许可文本
```

## 构建方法

### 环境要求

- macOS 14+，已安装 Xcode Command Line Tools
- Python 3.9+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 打包

```bash
./scripts/build-pack.sh
```

脚本会：

1. 从 Hugging Face 下载 Pix2Text-MFR 权重
2. 运行 `scripts/convert.py` 生成 Core ML FP16 模型并做数值对齐校验
3. 用 `xcrun coremlc` 预编译为 `.mlmodelc`
4. 在 `out/` 下生成：
   - `FormulaRecognitionMFR/` — 解压后的模型包
   - `FormulaRecognitionMFR.zip` / `FormulaRecognitionMFR.zip.sha256`
   - `FormulaRecognitionMFR.aar` — Background Assets 归档（工具链可用时）

构建成功后可做冒烟检查：

```bash
python3 scripts/smoke.py
```

## Swift 调用（Xmosome 客户端）

```swift
guard let modelDir = FormulaModelStore.shared.modelDirectory else { return }
let recognizer = try FormulaMFRRecognizer.shared(at: modelDir)
let result = try recognizer.recognize(image, source: .screenshot) { _ in }
let latex = result.latex
```

`FormulaModelStore` / `FormulaMFRRecognizer` 位于 Xmosome 应用工程中，不在本仓库。客户端查找合法模型目录（`pack.json`、两个 `.mlmodelc`、`vocab.json`）。

## 仓库结构

```text
.
├── README.md / README_CN.md
├── LICENSE
├── requirements.txt
├── .github/workflows/build-model-pack.yml
└── scripts/
    ├── build-pack.sh
    ├── convert.py
    ├── smoke.py
    └── AssetPackManifest.json
```

## 开源协议

- 上游 Pix2Text-MFR 1.5 — MIT（Copyright (c) 2022 BreezeDeus）；见包内 `LICENSE-Pix2Text.txt`
- 本仓库转换与打包脚本 — [MIT](LICENSE)
