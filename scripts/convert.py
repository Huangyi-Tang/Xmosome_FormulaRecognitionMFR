"""Convert Pix2Text-MFR 1.5 (MIT, breezedeus/pix2text-mfr-1.5) to Core ML.

The upstream repository publishes ONNX only. This script rebuilds the same
TrOCR VisionEncoderDecoder architecture with `transformers`, copies the ONNX
weights into it by module path, verifies the result against ONNX Runtime, and
exports two FP16 ML Programs:

  MFREncoder.mlpackage  pixel_values [1,3,384,384] -> encoder_hidden_states [1,578,384]
  MFRDecoder.mlpackage  input_ids [1,L] + encoder_hidden_states -> logits [1,L,1868],
                        L in DECODER_BUCKETS

plus `vocab.json` (id -> byte-level BPE token) for detokenization in Swift.

Usage (see README.md):
  python convert.py --onnx /tmp/mfr/onnx --out /tmp/mfr/pack
"""

import argparse
import json
import pathlib

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

MAX_DECODER_TOKENS = 256
# Core ML specializes a model per input shape; a free RangeDim recompiles at
# every new sequence length (~0.2 s each). Callers right-pad input_ids to the
# next bucket instead. The decoder is causal, so padding never changes the
# logits at earlier positions.
DECODER_BUCKETS = [32, 64, 128, 256]


def onnx_weights(path):
    """Return {torch_parameter_name: tensor} for one ONNX graph."""
    graph = onnx.load(str(path)).graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    weights = {}
    for name, value in initializers.items():
        if not name.startswith("onnx::"):
            weights[name] = value
    # Linear weights were exported as anonymous, transposed MatMul operands.
    # The consuming node's name carries the module path, e.g.
    # /encoder/layer.0/attention/attention/key/MatMul.
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        for operand in node.input:
            if operand.startswith("onnx::") and operand in initializers:
                module = node.name.removesuffix("/MatMul").strip("/").replace("/", ".")
                weights[module + ".weight"] = initializers[operand].T
    return weights


def load_model(onnx_dir):
    config = VisionEncoderDecoderConfig.from_pretrained(onnx_dir)
    model = VisionEncoderDecoderModel(config).eval()
    state = model.state_dict()

    loaded = {}
    for name, value in onnx_weights(onnx_dir / "encoder_model.onnx").items():
        # ONNX encoder graph is the bare DeiT model.
        loaded["encoder." + name] = value
    for name, value in onnx_weights(onnx_dir / "decoder_model.onnx").items():
        # Named weights already use the VisionEncoderDecoder prefix; MatMul
        # node paths are /decoder/decoder/layers.N/... and /decoder/output_projection.
        if name.startswith("decoder.decoder."):
            name = "decoder.model.decoder." + name[len("decoder.decoder."):]
        elif name.startswith("decoder.output_projection"):
            pass
        loaded[name] = value

    unexpected = sorted(set(loaded) - set(state))
    missing = sorted(set(state) - set(loaded))
    if unexpected:
        raise SystemExit(f"Unmapped ONNX weights: {unexpected[:10]}")
    # The DeiT pooler is unused by the decoder and absent from the export.
    missing = [m for m in missing if ".pooler." not in m]
    if missing:
        raise SystemExit(f"Missing weights: {missing[:10]}")
    for name, value in loaded.items():
        if tuple(state[name].shape) != value.shape:
            raise SystemExit(f"Shape mismatch for {name}: {state[name].shape} vs {value.shape}")
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in loaded.items()}, strict=False)
    return model


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state


class Decoder(torch.nn.Module):
    """Full-sequence decoder (no KV cache: Core ML state needs macOS 15)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, input_ids, encoder_hidden_states):
        return self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states).logits


def verify(model, onnx_dir):
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    pixels = rng.uniform(-1, 1, (1, 3, 384, 384)).astype(np.float32)
    ids = np.array([[1, 40, 41, 42, 43]], dtype=np.int64)
    encoder = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"))
    decoder = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"))
    hidden_ref = encoder.run(None, {"pixel_values": pixels})[0]
    logits_ref = decoder.run(None, {"input_ids": ids, "encoder_hidden_states": hidden_ref})[0]
    with torch.no_grad():
        hidden = Encoder(model)(torch.from_numpy(pixels)).numpy()
        logits = Decoder(model)(torch.from_numpy(ids), torch.from_numpy(hidden_ref)).numpy()[:, -1, :]
    print("encoder max |diff|", float(np.abs(hidden - hidden_ref).max()))
    print("decoder max |diff|", float(np.abs(logits - logits_ref[:, -1, :]).max()))
    assert np.abs(hidden - hidden_ref).max() < 1e-3
    assert np.abs(logits - logits_ref[:, -1, :]).max() < 1e-2


def export(model, out):
    import coremltools as ct

    out.mkdir(parents=True, exist_ok=True)
    pixels = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        traced_encoder = torch.jit.trace(Encoder(model).eval(), pixels)
        hidden = traced_encoder(pixels)
        ids = torch.ones(1, DECODER_BUCKETS[0], dtype=torch.int32)
        traced_decoder = torch.jit.trace(Decoder(model).eval(), (ids, hidden))

    encoder = ct.convert(
        traced_encoder,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 384, 384), dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_hidden_states", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    encoder.short_description = "Pix2Text-MFR 1.5 encoder (MIT, breezedeus)."
    encoder.save(str(out / "MFREncoder.mlpackage"))

    decoder = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.EnumeratedShapes(shapes=[(1, n) for n in DECODER_BUCKETS], default=(1, DECODER_BUCKETS[0])),
                dtype=np.int32,
            ),
            ct.TensorType(name="encoder_hidden_states", shape=tuple(hidden.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    decoder.short_description = "Pix2Text-MFR 1.5 decoder (MIT, breezedeus)."
    decoder.save(str(out / "MFRDecoder.mlpackage"))


def export_vocabulary(onnx_dir, out):
    tokenizer = json.loads((onnx_dir / "tokenizer.json").read_text())
    vocab = tokenizer["model"]["vocab"]
    tokens = [""] * (max(vocab.values()) + 1)
    for token, index in vocab.items():
        tokens[index] = token
    generation = json.loads((onnx_dir / "generation_config.json").read_text())
    (out / "vocab.json").write_text(json.dumps({
        "tokens": tokens,
        "decoder_start_token_id": generation["decoder_start_token_id"],
        "eos_token_id": generation["eos_token_id"],
        "pad_token_id": generation["pad_token_id"],
        "max_tokens": MAX_DECODER_TOKENS,
        "decoder_buckets": DECODER_BUCKETS,
    }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    model = load_model(args.onnx)
    verify(model, args.onnx)
    export(model, args.out)
    export_vocabulary(args.onnx, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
