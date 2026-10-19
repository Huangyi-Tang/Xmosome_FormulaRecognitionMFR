"""Run the converted Core ML MFR models on formula images and time them.

  python smoke.py --pack /tmp/mfr/pack image.png [image.png ...]
"""

import argparse
import json
import pathlib
import time

import coremltools as ct
import numpy as np
from PIL import Image


def byte_decoder():
    # GPT-2 byte-level BPE: printable bytes map to themselves, others to U+0100+.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def preprocess(image):
    # Pad to a white square first: stretching wide formulas to 384x384 makes
    # the model emit redundant brace groups.
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    image = canvas
    image = image.resize((384, 384), Image.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    return pixels.transpose(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=pathlib.Path, required=True)
    parser.add_argument("images", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    vocab = json.loads((args.pack / "vocab.json").read_text())
    tokens, decode = vocab["tokens"], byte_decoder()
    encoder = ct.models.MLModel(str(args.pack / "MFREncoder.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    # The bucketed decoder crashes on the Neural Engine, and for this small
    # model the CPU is also the fastest option (no dispatch overhead per token).
    decoder = ct.models.MLModel(str(args.pack / "MFRDecoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)

    for path in args.images:
        start = time.perf_counter()
        hidden = encoder.predict({"pixel_values": preprocess(Image.open(path))})["encoder_hidden_states"]
        encoded = time.perf_counter()
        ids = [vocab["decoder_start_token_id"]]
        while len(ids) < vocab["max_tokens"]:
            bucket = next(n for n in vocab["decoder_buckets"] if n >= len(ids))
            padded = ids + [vocab["pad_token_id"]] * (bucket - len(ids))
            logits = decoder.predict({"input_ids": np.array([padded], dtype=np.int32), "encoder_hidden_states": hidden})["logits"]
            token = int(np.argmax(logits[0, len(ids) - 1]))
            if token == vocab["eos_token_id"]:
                break
            ids.append(token)
        done = time.perf_counter()
        text = bytes(b for t in ids[1:] for ch in tokens[t] for b in [decode.get(ch, 63)]).decode("utf-8", "replace")
        print(f"{path.name}: {len(ids) - 1} tokens, encode {encoded - start:.2f}s, decode {done - encoded:.2f}s")
        print("   ", text.strip())


if __name__ == "__main__":
    main()
