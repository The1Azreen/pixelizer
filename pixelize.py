"""Turn any image into crisp retro pixel art, icons and sprites.

    python pixelize.py photo.jpg                     # -> photo_32x32.png
    python pixelize.py photo.jpg -s 64x48 --fit crop
    python pixelize.py logo.png -s 16 --palette pico-8 -o logo.ico
    python pixelize.py cat.jpg -s 48 --remove-bg --outline --scale 8

How it works: colours are clustered in OKLab (a perceptual colour space) to build
the palette, then every output pixel takes the colour that *most* of its source
area belongs to (instead of the average), which keeps edges sharp rather than
muddy. Optional dithering, outlines and AI background removal finish the look.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

# When packaged as an .exe, keep models/ next to the exe rather than in the temp unpack dir.
ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
MODELS = ROOT / "models"
CUTOUT_MODEL = MODELS / "u2netp.onnx"

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".ico")
SAVE_TYPES = (".png", ".ico", ".gif", ".webp", ".bmp")
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _hex(*codes):
    return [tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for c in codes]


PALETTES = {
    "pico-8": _hex("000000", "1d2b53", "7e2553", "008751", "ab5236", "5f574f", "c2c3c7", "fff1e8",
                   "ff004d", "ffa300", "ffec27", "00e436", "29adff", "83769c", "ff77a8", "ffccaa"),
    "sweetie-16": _hex("1a1c2c", "5d275d", "b13e53", "ef7d57", "ffcd75", "a7f070", "38b764", "257179",
                       "29366f", "3b5dc9", "41a6f6", "73eff7", "f4f4f4", "94b0c2", "566c86", "333c57"),
    "endesga-32": _hex("be4a2f", "d77643", "ead4aa", "e4a672", "b86f50", "733e39", "3e2731", "a22633",
                       "e43b44", "f77622", "feae34", "fee761", "63c74d", "3e8948", "265c42", "193c3e",
                       "124e89", "0099db", "2ce8f5", "ffffff", "c0cbdc", "8b9bb4", "5a6988", "3a4466",
                       "262b44", "181425", "ff0044", "68386c", "b55088", "f6757a", "e8b796", "c28569"),
    "c64": _hex("000000", "ffffff", "68372b", "70a4b2", "6f3d86", "588d43", "352879", "b8c76f",
                "6f4f25", "433900", "9a6759", "444444", "6c6c6c", "9ad284", "6c5eb5", "959595"),
    "game-boy": _hex("0f380f", "306230", "8bac0f", "9bbc0f"),
    "cga": _hex("000000", "55ffff", "ff55ff", "ffffff"),
    "1-bit": _hex("000000", "ffffff"),
}

FITS = ("fit", "crop", "stretch")
DITHERS = ("none", "ordered", "floyd")
METHODS = ("smart", "average", "nearest")


@dataclass
class Settings:
    width: int = 32           # output width in pixels (0 = from height / aspect)
    height: int = 0           # output height in pixels (0 = from width / aspect)
    pixel_size: int = 0       # alternative to width/height: source pixels per output pixel
    fit: str = "fit"          # how to handle a different aspect ratio: fit (pad), crop, stretch
    palette: str = "auto"     # "auto" or a key of PALETTES
    colors: int = 16          # palette size for "auto"
    dither: str = "none"      # none, ordered (Bayer) or floyd (Floyd-Steinberg)
    method: str = "smart"     # smart (majority colour), average or nearest
    outline: bool = False     # 1px outline around the opaque shape
    remove_bg: bool = False   # AI background removal (needs onnxruntime + model)
    trim: bool = True         # crop away fully transparent borders first
    cleanup: bool = True      # remove stray lone pixels and cutout specks (pixels skipped when dithering)
    contrast: float = 1.0
    saturation: float = 1.0
    background: str = ""      # hex fill colour; blank keeps transparency


# ---------- colour ----------

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def to_oklab(rgb):
    """sRGB 0-255 (..., 3) -> OKLab."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(c @ _M1.T) @ _M2.T


def from_oklab(lab):
    """OKLab (..., 3) -> sRGB 0-255 floats."""
    c = ((np.asarray(lab) @ _M2_INV.T) ** 3) @ _M1_INV.T
    c = np.clip(c, 0, 1)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def parse_hex(text):
    text = text.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Not a hex colour: #{text}")
    return _hex(text)[0]


def nearest(lab, pal_lab, chunk=200_000):
    """Index of the nearest palette colour for each row of lab (N, 3)."""
    out = np.empty(len(lab), dtype=np.int32)
    for i in range(0, len(lab), chunk):
        part = lab[i:i + chunk]
        d = ((part[:, None, :] - pal_lab[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def kmeans_palette(lab, k, iters=16, sample=20_000, seed=0):
    """Cluster OKLab colours into k centres (k-means++ init, deterministic)."""
    rng = np.random.default_rng(seed)
    if len(lab) > sample:
        lab = lab[rng.choice(len(lab), sample, replace=False)]
    uniq = np.unique(np.round(lab, 4), axis=0)
    if len(uniq) <= k:
        return uniq
    centres = [lab[rng.integers(len(lab))]]
    d2 = ((lab - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        centres.append(lab[rng.choice(len(lab), p=d2 / d2.sum())] if d2.sum() > 0 else lab[rng.integers(len(lab))])
        d2 = np.minimum(d2, ((lab - centres[-1]) ** 2).sum(1))
    centres = np.array(centres)
    for _ in range(iters):
        labels = nearest(lab, centres)
        counts = np.bincount(labels, minlength=k)
        sums = np.stack([np.bincount(labels, weights=lab[:, c], minlength=k) for c in range(3)], 1)
        empty = counts == 0
        centres[~empty] = sums[~empty] / counts[~empty, None]
        if empty.any():  # re-seed empty clusters on the worst-fitting points
            far = ((lab - centres[labels]) ** 2).sum(1).argsort()[::-1]
            centres[empty] = lab[far[:empty.sum()]]
    return centres


# ---------- AI background removal ----------

_session = None


def cutout_status():
    """(available, message) for the AI background remover."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "AI cutout needs onnxruntime (pip install onnxruntime)"
    if not CUTOUT_MODEL.is_file():
        return False, "AI cutout model not installed"
    return True, f"AI cutout ready · {cutout_device()}"


def cutout_device():
    try:
        import onnxruntime as ort
    except ImportError:
        return "unavailable"
    providers = ort.get_available_providers()
    if "DmlExecutionProvider" in providers:
        return "GPU (DirectML)"
    if "CUDAExecutionProvider" in providers:
        return "GPU (CUDA)"
    if "CoreMLExecutionProvider" in providers:
        return "GPU (CoreML)"
    return "CPU"


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not CUTOUT_MODEL.is_file():
            raise RuntimeError("AI model missing - run `python setup_tools.py` first.")
        wanted = ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
        available = ort.get_available_providers()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(str(CUTOUT_MODEL), opts, providers=[p for p in wanted if p in available])
    return _session


def remove_background(img):
    """Return img (RGBA) with the background made transparent by U^2-Net-p (320x320 saliency model)."""
    session = _get_session()
    rgb = img.convert("RGB")
    x = np.asarray(rgb.resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
    x /= max(float(x.max()), 1.0)
    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    pred = session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.Resampling.LANCZOS)
    out = img.convert("RGBA")
    alpha = np.minimum(np.asarray(out.getchannel("A")), np.asarray(mask))
    out.putalpha(Image.fromarray(alpha))
    return out


# ---------- pipeline ----------

def plan(src_w, src_h, s, margin=0):
    """Work out (canvas size, content size, source crop box) for the settings."""
    box = (0, 0, src_w, src_h)
    if s.pixel_size and not (s.width or s.height):
        cw, ch = max(1, round(src_w / s.pixel_size)), max(1, round(src_h / s.pixel_size))
        return (cw + 2 * margin, ch + 2 * margin), (cw, ch), box
    w, h = max(0, s.width), max(0, s.height)
    if not w and not h:
        w = 32
    if w and h:
        iw, ih = max(1, w - 2 * margin), max(1, h - 2 * margin)
        if s.fit == "stretch":
            return (w, h), (iw, ih), box
        if s.fit == "crop":
            target = iw / ih
            if src_w / src_h > target:
                cw = src_h * target
                box = (round((src_w - cw) / 2), 0, round((src_w + cw) / 2), src_h)
            else:
                ch = src_w / target
                box = (0, round((src_h - ch) / 2), src_w, round((src_h + ch) / 2))
            return (w, h), (iw, ih), box
        scale = min(iw / src_w, ih / src_h)
        return (w, h), (max(1, round(src_w * scale)), max(1, round(src_h * scale))), box
    if w:
        iw = max(1, w - 2 * margin)
        ih = max(1, round(src_h * iw / src_w))
    else:
        ih = max(1, h - 2 * margin)
        iw = max(1, round(src_w * ih / src_h))
    return (iw + 2 * margin, ih + 2 * margin), (iw, ih), box


def prepare(img, s, cutout=None):
    """Load-time steps shared by every render: RGBA, AI cutout, trim and colour tweaks."""
    img = img.convert("RGBA")
    if s.remove_bg:
        img = cutout if cutout is not None else remove_background(img)
    if s.trim:
        bbox = img.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
        if bbox and bbox != (0, 0, *img.size):
            img = img.crop(bbox)
    if s.contrast != 1.0 or s.saturation != 1.0:
        alpha = img.getchannel("A")
        rgb = img.convert("RGB")
        if s.contrast != 1.0:
            rgb = ImageEnhance.Contrast(rgb).enhance(s.contrast)
        if s.saturation != 1.0:
            rgb = ImageEnhance.Color(rgb).enhance(s.saturation)
        img = rgb.convert("RGBA")
        img.putalpha(alpha)
    return img


def downscale(hi, k, size, pal_lab, method):
    """Reduce hi (H*k, W*k RGBA array) to (H, W) OKLab colours + opaque mask."""
    w, h = size
    rgb = hi[..., :3].astype(np.float64)
    opaque_hi = hi[..., 3] >= 128
    cover = opaque_hi.reshape(h, k, w, k).mean((1, 3))
    opaque = cover >= 0.5

    if method == "nearest":
        c = k // 2
        lab = to_oklab(rgb[c::k, c::k])
        return lab, opaque_hi[c::k, c::k]

    lab_hi = to_oklab(rgb)
    if method == "average":
        weights = opaque_hi.reshape(h, k, w, k).astype(np.float64)
        weights = np.where(weights.sum((1, 3), keepdims=True) > 0, weights, 1.0)
        blocks = lab_hi.reshape(h, k, w, k, 3)
        lab = (blocks * weights[..., None]).sum((1, 3)) / weights.sum((1, 3))[..., None]
        return lab, opaque

    # smart: per block, find the palette colour most of the (opaque) subpixels belong to,
    # then use the mean of just those subpixels so the colour is true but never a blend of an edge.
    p = len(pal_lab)
    labels = nearest(lab_hi.reshape(-1, 3), pal_lab).reshape(h * k, w * k)
    block = (np.arange(h)[:, None, None, None] * w + np.arange(w)[None, None, :, None])
    block = np.broadcast_to(block, (h, k, w, k)).reshape(h * k, w * k)
    use = opaque_hi | ~opaque.repeat(k, 0).repeat(k, 1)  # transparent blocks still get a colour
    idx = (block * p + labels)[use]
    counts = np.bincount(idx, minlength=h * w * p).reshape(h * w, p)
    sums = np.stack([np.bincount(idx, weights=lab_hi[..., c][use], minlength=h * w * p) for c in range(3)], -1)
    sums = sums.reshape(h * w, p, 3)
    best = counts.argmax(1)
    rows = np.arange(h * w)
    lab = sums[rows, best] / np.maximum(counts[rows, best], 1)[:, None]
    return lab.reshape(h, w, 3), opaque


BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32


def quantize(lab, opaque, pal_lab, dither):
    """Map OKLab pixels (H, W, 3) to palette indices with optional dithering."""
    h, w, _ = lab.shape
    if dither == "ordered":
        # Scale the pattern to the typical gap between neighbouring palette colours,
        # so dense palettes get a subtle dither and 2-4 colour palettes a strong one.
        pal_rgb = from_oklab(pal_lab)
        gaps = np.sqrt(((pal_rgb[:, None] - pal_rgb[None]) ** 2).sum(-1))
        np.fill_diagonal(gaps, np.inf)
        spread = float(np.median(gaps.min(1))) * 0.9 if len(pal_lab) > 1 else 0.0
        rgb = from_oklab(lab)
        offset = (np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w] - 0.5) * spread
        lab = to_oklab(np.clip(rgb + offset[..., None], 0, 255))
        return nearest(lab.reshape(-1, 3), pal_lab).reshape(h, w)
    if dither == "floyd":
        work = lab.copy()
        out = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                old = work[y, x]
                i = int(((pal_lab - old) ** 2).sum(1).argmin())
                out[y, x] = i
                if not opaque[y, x]:
                    continue
                err = (old - pal_lab[i]) * 0.75
                if x + 1 < w and opaque[y, x + 1]:
                    work[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x > 0 and opaque[y + 1, x - 1]:
                        work[y + 1, x - 1] += err * 3 / 16
                    if opaque[y + 1, x]:
                        work[y + 1, x] += err * 5 / 16
                    if x + 1 < w and opaque[y + 1, x + 1]:
                        work[y + 1, x + 1] += err * 1 / 16
        return out
    return nearest(lab.reshape(-1, 3), pal_lab).reshape(h, w)


def clean_strays(idx, opaque, passes=1):
    """Replace lone pixels (no 4-neighbour shares their colour) with their most common neighbour colour."""
    h, w = idx.shape
    for _ in range(passes):
        pad = np.pad(np.where(opaque, idx, -1), 1, constant_values=-1)
        neigh = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        lonely = opaque & ~(neigh == idx).any(0)
        if not lonely.any():
            break
        ys, xs = np.nonzero(lonely)
        votes = neigh[:, ys, xs].T  # (n, 4)
        for j, (y, x) in enumerate(zip(ys, xs)):
            v = votes[j][votes[j] >= 0]
            if len(v) >= 3:  # only where the pixel is clearly surrounded
                vals, counts = np.unique(v, return_counts=True)
                if counts.max() >= 2:
                    idx[y, x] = vals[counts.argmax()]
    return idx


def drop_islands(opaque, keep=0.08):
    """Clear opaque blobs smaller than keep * the largest blob (stray specks left by a cutout)."""
    h, w = opaque.shape
    label = np.zeros((h, w), dtype=np.int32)
    sizes = [0]
    for y0, x0 in zip(*np.nonzero(opaque)):
        if label[y0, x0]:
            continue
        n = len(sizes)
        label[y0, x0] = n
        stack, size = [(y0, x0)], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and not label[ny, nx]:
                    label[ny, nx] = n
                    stack.append((ny, nx))
        sizes.append(size)
    sizes = np.array(sizes)
    if len(sizes) <= 2:
        return opaque
    return opaque & (sizes[label] >= keep * sizes.max())


def pixelize(img, s=None, cutout=None):
    """Pixelize a PIL image. Returns (RGBA image at native size, list of palette RGB tuples).

    cutout: optional pre-computed remove_background(img) result, so callers can cache it.
    """
    s = s or Settings()
    img = prepare(img, s, cutout)
    margin = 1 if s.outline else 0
    canvas, (cw, ch), box = plan(*img.size, s, margin)
    if box != (0, 0, *img.size):
        img = img.crop(box)

    # Supersample factor: enough subpixels per output pixel for a good vote, capped for speed.
    k = int(np.clip(min(img.width / cw, img.height / ch), 1, 8))
    k = max(1, min(k, int((1_500_000 / (cw * ch)) ** 0.5)))
    hi = img.resize((cw * k, ch * k), Image.Resampling.LANCZOS if k > 1 else Image.Resampling.BOX)
    hi = np.asarray(hi)
    # Premultiply-ish: colour under fully transparent pixels is often junk; don't let it vote.
    opaque_hi = hi[..., 3] >= 128

    if s.palette in PALETTES:
        pal_rgb = np.array(PALETTES[s.palette], dtype=np.float64)
        pal_lab = to_oklab(pal_rgb)
    else:
        source = hi[..., :3][opaque_hi] if opaque_hi.any() else hi[..., :3].reshape(-1, 3)
        pal_lab = kmeans_palette(to_oklab(source.astype(np.float64)), max(2, min(int(s.colors), 256)))
        pal_rgb = np.round(from_oklab(pal_lab))
        pal_lab = to_oklab(pal_rgb)

    lab, opaque = downscale(hi, k, (cw, ch), pal_lab, s.method)
    if s.cleanup and s.remove_bg:
        opaque = drop_islands(opaque)
    idx = quantize(lab, opaque, pal_lab, s.dither)
    if s.cleanup and s.dither == "none":
        idx = clean_strays(idx, opaque)

    rgba = np.zeros((canvas[1], canvas[0], 4), dtype=np.uint8)
    ox, oy = (canvas[0] - cw) // 2, (canvas[1] - ch) // 2
    region = rgba[oy:oy + ch, ox:ox + cw]
    region[..., :3] = pal_rgb[idx].astype(np.uint8)
    region[..., 3] = np.where(opaque, 255, 0)

    if s.outline:
        add_outline(rgba, pal_rgb[pal_lab[:, 0].argmin()].astype(np.uint8))

    used = np.unique(idx[opaque]) if opaque.any() else np.array([], dtype=int)
    palette = [tuple(int(v) for v in pal_rgb[i]) for i in used]
    if s.outline and opaque.any():
        dark = tuple(int(v) for v in pal_rgb[pal_lab[:, 0].argmin()])
        if dark not in palette:
            palette.append(dark)

    out = Image.fromarray(rgba, "RGBA")
    if s.background:
        bg = Image.new("RGBA", out.size, (*parse_hex(s.background), 255))
        out = Image.alpha_composite(bg, out)
    palette.sort(key=lambda c: float(to_oklab(np.array(c, dtype=np.float64))[0]))
    return out, palette


def add_outline(rgba, color):
    """Paint transparent pixels that touch the opaque shape (4-neighbourhood) with color, in place."""
    solid = rgba[..., 3] > 0
    if solid.all() or not solid.any():
        return
    ring = np.zeros_like(solid)
    ring[1:] |= solid[:-1]
    ring[:-1] |= solid[1:]
    ring[:, 1:] |= solid[:, :-1]
    ring[:, :-1] |= solid[:, 1:]
    ring &= ~solid
    rgba[ring, :3] = color
    rgba[ring, 3] = 255


# ---------- saving ----------

def upscale(img, scale):
    return img if scale <= 1 else img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def save(img, path, scale=1):
    """Save a pixelized image. The extension picks the format; scale enlarges with hard pixel edges."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SAVE_TYPES:
        raise ValueError(f"Unsupported output type {ext!r} (use {', '.join(SAVE_TYPES)})")
    if ext == ".ico":
        # Icons must be square; pad, then give every standard size its own nearest-neighbour copy
        # so Windows never blurs the pixels when it picks a size.
        side = max(img.size)
        square = Image.new("RGBA", (side, side))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        sizes = sorted({min(side, 256), *ICO_SIZES})
        images = [square.resize((n, n), Image.Resampling.NEAREST) for n in sizes]
        images[-1].save(path, format="ICO", sizes=[(n, n) for n in sizes], append_images=images[:-1])
        return path
    big = upscale(img, scale)
    if ext == ".gif":
        # Pixel art has few colours, so a 255-colour quantize is exact; index 255 is kept for transparency.
        clear = np.asarray(big.getchannel("A")) < 128
        p = big.convert("RGB").quantize(colors=255, dither=Image.Dither.NONE)
        if clear.any():
            data = np.asarray(p).copy()
            data[clear] = 255
            pal = p.getpalette()[:765]
            q = Image.fromarray(data, "P")
            q.putpalette(pal + [0] * (768 - len(pal)))
            q.save(path, transparency=255)
        else:
            p.save(path)
    elif ext == ".webp":
        big.save(path, lossless=True)
    else:
        big.save(path)
    return path


def default_output(src, img, ext=".png"):
    src = Path(src)
    return src.with_name(f"{src.stem}_{img.width}x{img.height}{ext}")


# ---------- CLI ----------

def parse_size(text):
    text = text.lower().replace(" ", "")
    if "x" in text:
        w, h = text.split("x", 1)
        return int(w or 0), int(h or 0)
    n = int(text)
    return n, n


def build_parser():
    p = argparse.ArgumentParser(
        description="Turn an image into retro pixel art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="palettes: auto, " + ", ".join(PALETTES))
    p.add_argument("input", help="Image file")
    p.add_argument("-o", "--output", help="Output file; extension picks the format "
                                          "(.png .ico .gif .webp .bmp). Default: <input>_<W>x<H>.png")
    p.add_argument("-s", "--size",
                   help="Output size: N (square), WxH, Wx (keep aspect) or xH. Default 32")
    p.add_argument("-p", "--pixel-size", type=int, default=0,
                   help="Instead of --size: source pixels per output pixel (keeps original aspect)")
    p.add_argument("--fit", choices=FITS, default="fit",
                   help="When the aspect differs: fit (pad), crop or stretch. Default fit")
    p.add_argument("--palette", default="auto", choices=["auto", *PALETTES], help="Colour palette. Default auto")
    p.add_argument("-c", "--colors", type=int, default=16, help="Colours for the auto palette. Default 16")
    p.add_argument("-d", "--dither", choices=DITHERS, default="none", help="Dithering. Default none")
    p.add_argument("-m", "--method", choices=METHODS, default="smart", help="Downscale method. Default smart")
    p.add_argument("--outline", action="store_true", help="Add a 1px dark outline around the shape")
    p.add_argument("--remove-bg", action="store_true", help="AI background removal (see setup_tools.py)")
    p.add_argument("--no-trim", action="store_true", help="Keep transparent borders")
    p.add_argument("--no-cleanup", action="store_true", help="Keep stray single pixels")
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast boost, 1 = unchanged")
    p.add_argument("--saturation", type=float, default=1.0, help="Saturation boost, 1 = unchanged")
    p.add_argument("--background", default="", help="Fill transparency with this hex colour, e.g. 1a1c2c")
    p.add_argument("--scale", type=int, default=1, help="Enlarge the saved image N times with hard pixel edges")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"Error: {src} not found")
    width, height = parse_size(args.size) if args.size else ((0, 0) if args.pixel_size else (32, 32))
    s = Settings(width=width, height=height, pixel_size=args.pixel_size, fit=args.fit, palette=args.palette,
                 colors=args.colors, dither=args.dither, method=args.method, outline=args.outline,
                 remove_bg=args.remove_bg, trim=not args.no_trim, cleanup=not args.no_cleanup,
                 contrast=args.contrast,
                 saturation=args.saturation, background=args.background)
    if s.remove_bg:
        ok, msg = cutout_status()
        if not ok:
            sys.exit(f"Error: {msg}")
    t = time.perf_counter()
    with Image.open(src) as im:
        im.load()
        result, palette = pixelize(im, s)
    out = Path(args.output) if args.output else default_output(src, result)
    save(result, out, max(1, args.scale))
    ms = (time.perf_counter() - t) * 1000
    print(f"{result.width}x{result.height}, {len(palette)} colours, {ms:.0f} ms -> {out}")


if __name__ == "__main__":
    main()
