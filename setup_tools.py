"""Download the AI background-removal model (U^2-Net-p, 4.6 MB) into ./models.

It runs through onnxruntime: on the CPU by default, or on the GPU if you install
onnxruntime-directml (Windows, any DX12 GPU) or onnxruntime-gpu (NVIDIA CUDA).
"""

import sys
import urllib.request

from pixelize import CUTOUT_MODEL

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"


def main(progress=None):
    """progress: optional callback(fraction) for the GUI."""
    if CUTOUT_MODEL.is_file():
        print(f"AI model already installed: {CUTOUT_MODEL}")
        return
    print(f"Downloading {URL}")
    CUTOUT_MODEL.parent.mkdir(exist_ok=True)
    part = CUTOUT_MODEL.with_suffix(".part")
    with urllib.request.urlopen(URL) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    part.replace(CUTOUT_MODEL)
    print(f"Installed AI model to {CUTOUT_MODEL}")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("Note: also run `pip install onnxruntime` (or onnxruntime-directml for GPU).", file=sys.stderr)


if __name__ == "__main__":
    main()
