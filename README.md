# Xmosome_FormulaRecognitionMFR

[English](README.md) | [简体中文](README_CN.md)

Offline conversion pipeline that turns [Pix2Text-MFR 1.5](https://huggingface.co/breezedeus/pix2text-mfr-1.5) into Apple Core ML (FP16) packs for Xmosome.

Use cases in the Xmosome apps:

- macOS — recognize formulas from screen captures
- iPadOS / iOS — recognize Apple Pencil handwriting as LaTeX

Inference runs entirely on device (Neural Engine / GPU). This repository only builds and packages the model; the client apps load the resulting pack at runtime.

## Architecture

| Piece | Detail |
| --- | --- |
| Upstream model | [breezedeus/pix2text-mfr-1.5](https://huggingface.co/breezedeus/pix2text-mfr-1.5) — TrOCR-style DeiT-small encoder + causal decoder, vocab 1868, MIT |
| Runtime | Core ML 8.x `mlprogram`, FP16 |
| Sequence bucketing | Decoder lengths `[32, 64, 128, 256]` to avoid dynamic-shape recompiles |
| Upstream release format | Hugging Face ships ONNX weights; packaging converts them to Core ML. ONNX is not used at app runtime |

## Model pack layout

`./scripts/build-pack.sh` writes a folder with this layout (also zipped for distribution):

```text
FormulaRecognitionMFR/
├── pack.json             # Manifest (format version 1)
├── MFREncoder.mlmodelc/  # Vision encoder  [1, 3, 384, 384] → [1, 578, 384]
├── MFRDecoder.mlmodelc/  # Sequence decoder [1, L] + encoder states → [1, L, 1868]
├── vocab.json            # Byte-level BPE vocab and bucket config
└── LICENSE-Pix2Text.txt  # Upstream MIT license text
```

## Build

### Requirements

- macOS 14+ with Xcode Command Line Tools
- Python 3.9+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Package

```bash
./scripts/build-pack.sh
```

The script:

1. Downloads Pix2Text-MFR weights from Hugging Face
2. Runs `scripts/convert.py` to produce Core ML FP16 models and checks numerical parity
3. Compiles `.mlmodelc` with `xcrun coremlc`
4. Writes artifacts under `out/`:
   - `FormulaRecognitionMFR/` — unpacked pack
   - `FormulaRecognitionMFR.zip` / `FormulaRecognitionMFR.zip.sha256`
   - `FormulaRecognitionMFR.aar` — Background Assets archive (when tooling is available)

Optional smoke check after a successful build:

```bash
python3 scripts/smoke.py
```

## Swift usage (Xmosome client)

```swift
guard let modelDir = FormulaModelStore.shared.modelDirectory else { return }
let recognizer = try FormulaMFRRecognizer.shared(at: modelDir)
let result = try recognizer.recognize(image, source: .screenshot) { _ in }
let latex = result.latex
```

`FormulaModelStore` / `FormulaMFRRecognizer` live in the Xmosome app targets, not in this repo. The app looks for a valid pack directory (`pack.json` + both `.mlmodelc` bundles + `vocab.json`).

## Repository layout

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

## License

- Upstream Pix2Text-MFR 1.5 — MIT (Copyright (c) 2022 BreezeDeus); see packed `LICENSE-Pix2Text.txt`
- Conversion and packaging scripts in this repository — [MIT](LICENSE)
