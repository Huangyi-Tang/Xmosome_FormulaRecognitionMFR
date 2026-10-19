#!/bin/bash
# Builds the optional Pix2Text-MFR formula-recognition model pack.
#
#   TesTools/mfr-coreml/build-pack.sh [work-dir]
#
# Outputs in <work-dir>/out:
#   FormulaRecognitionMFR.aar  Apple-hosted Background Assets pack (App Store Connect)
#   FormulaRecognitionMFR.zip  Same folder for the direct-download build
#   FormulaRecognitionMFR.zip.sha256
#
# Requires a Python 3.9+ venv with: torch==2.5.1 transformers==4.46.3
# coremltools==8.1 onnx onnxruntime numpy pillow (see README.md).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
work="${1:-/tmp/mfr}"
revision="1cef9f0bdcd6a4c63df7de1311fb0894593340cc"
pack_id="FormulaRecognitionMFR"

mkdir -p "$work/onnx" "$work/out"
for file in config.json generation_config.json tokenizer.json encoder_model.onnx decoder_model.onnx; do
  [[ -f "$work/onnx/$file" ]] || curl -fsSL --retry 5 -o "$work/onnx/$file" \
    "https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/$revision/$file"
done

rm -rf "$work/pack" "$work/assetpack"
python "$here/convert.py" --onnx "$work/onnx" --out "$work/pack"

folder="$work/assetpack/$pack_id"
mkdir -p "$folder"
# Ship compiled models so the app never compiles Core ML at runtime.
xcrun coremlc compile "$work/pack/MFREncoder.mlpackage" "$folder" >/dev/null
xcrun coremlc compile "$work/pack/MFRDecoder.mlpackage" "$folder" >/dev/null
cp "$work/pack/vocab.json" "$folder/"
curl -fsSL --retry 3 -o "$folder/LICENSE-Pix2Text.txt" \
  https://raw.githubusercontent.com/breezedeus/Pix2Text/main/LICENSE
cat > "$folder/pack.json" <<EOF
{
  "format": 1,
  "model": "Pix2Text-MFR 1.5",
  "source": "https://huggingface.co/breezedeus/pix2text-mfr-1.5",
  "revision": "$revision",
  "license": "MIT (Copyright (c) 2022 BreezeDeus), see LICENSE-Pix2Text.txt",
  "conversion": "TesTools/mfr-coreml/convert.py (ONNX weights -> Core ML FP16)",
  "encoder": "MFREncoder.mlmodelc",
  "decoder": "MFRDecoder.mlmodelc",
  "vocabulary": "vocab.json"
}
EOF

(cd "$work/assetpack" && xcrun ba-package "$here/AssetPackManifest.json" -o "$work/out/$pack_id.aar")
rm -f "$work/out/$pack_id.zip"
(cd "$work/assetpack" && ditto -c -k --keepParent "$pack_id" "$work/out/$pack_id.zip")
shasum -a 256 "$work/out/$pack_id.zip" | cut -d' ' -f1 > "$work/out/$pack_id.zip.sha256"

ls -la "$work/out"
echo "zip sha256: $(cat "$work/out/$pack_id.zip.sha256")"
