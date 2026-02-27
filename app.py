import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI(title="Chroma Key Cleaner (Low-Memory)")
templates = Jinja2Templates(directory="templates")


# ----------------------------
# LUTs for sRGB <-> linear (memory & speed friendly)
# ----------------------------
_SRGB_TO_LINEAR_LUT = None
def srgb_to_linear_lut(dtype=np.float16) -> np.ndarray:
    global _SRGB_TO_LINEAR_LUT
    if _SRGB_TO_LINEAR_LUT is None:
        x = np.linspace(0, 1, 256, dtype=np.float32)
        a = 0.055
        lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
        _SRGB_TO_LINEAR_LUT = lin
    return _SRGB_TO_LINEAR_LUT.astype(dtype, copy=False)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    # x: float in [0,1]
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


# ----------------------------
# PIL-based morphology (much lower peak memory than np.pad)
# ----------------------------
def _bool_to_L(mask: np.ndarray) -> Image.Image:
    # mask bool -> 'L' 0/255
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _L_to_bool(imgL: Image.Image) -> np.ndarray:
    # 'L' -> bool
    return (np.asarray(imgL, dtype=np.uint8) > 127)


def dilate_bool(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    # MaxFilter size must be odd. Radius px -> size = 2*px+1
    size = 2 * px + 1
    img = _bool_to_L(mask)
    img2 = img.filter(ImageFilter.MaxFilter(size=size))
    return _L_to_bool(img2)


def erode_bool(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    size = 2 * px + 1
    img = _bool_to_L(mask)
    img2 = img.filter(ImageFilter.MinFilter(size=size))
    return _L_to_bool(img2)


def edge_band_from_alpha(alpha: np.ndarray, band_px: int) -> np.ndarray:
    """
    alpha: float [H,W] in [0,1]
    Edge band = pixels near alpha transition.
    Uses PIL morphology for low memory.
    """
    fg = alpha > 0.5
    d1 = dilate_bool(fg, 1)
    e1 = erode_bool(fg, 1)
    edge = d1 & (~e1)
    # Expand edge to band
    if band_px > 1:
        edge = dilate_bool(edge, band_px - 1)
    return edge


# ----------------------------
# Core pipeline
# ----------------------------
@dataclass
class Params:
    green_tolerance: int = 40      # 0..255
    edge_band_px: int = 4          # 1..12
    spill_strength: int = 65       # 0..100
    feather: float = 0.6           # 0..2
    alpha_clamp_low: float = 0.03  # 0..0.2
    alpha_clamp_high: float = 0.98 # 0.8..1
    hard_alpha: bool = True


def compute_chroma_alpha_linear(r: np.ndarray, g: np.ndarray, b: np.ndarray, tol: int) -> np.ndarray:
    """
    Aggressive chroma alpha in linear space.
    r,g,b: float arrays [H,W] linear 0..1 (float16/float32)
    Returns alpha float32 [H,W]
    """
    # Promote to float32 for stable math without huge extra copies
    r32 = r.astype(np.float32, copy=False)
    g32 = g.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)

    t = tol / 255.0
    dom = g32 - np.maximum(r32, b32)
    bg_score = dom * (g32 + 0.15)

    cutoff = 0.08 - (t * 0.06)
    softness = 0.03 + (t * 0.02)

    x = (bg_score - cutoff) / max(1e-6, softness)
    x = np.clip(x, 0.0, 1.0)
    s = x * x * (3 - 2 * x)  # smoothstep
    alpha = 1.0 - s
    return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)


def spill_suppress_edge_only_inplace(r: np.ndarray, g: np.ndarray, b: np.ndarray, edge_mask: np.ndarray, strength: int):
    """
    In-place spill suppression (only modifies g) to avoid big copies.
    r,g,b: linear float16 arrays [H,W]
    edge_mask: bool [H,W]
    """
    if strength <= 0:
        return

    k = strength / 100.0

    # Work in float32 views for math; store back to float16 g
    r32 = r.astype(np.float32, copy=False)
    g32 = g.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)

    maxrb = np.maximum(r32, b32)
    green_dom = g32 > (maxrb + 1e-6)
    m = edge_mask & green_dom
    if not np.any(m):
        return

    target_g = (np.minimum(r32, b32) * 0.8) + ((r32 + b32) * 0.5 * 0.2)
    g_new = (1.0 - k) * g32 + k * target_g

    # Write back only where mask is true (minimizes temporary allocations)
    g_out = g32.copy()  # small-ish overhead but avoids building full np.where result
    g_out[m] = g_new[m]
    np.clip(g_out, 0.0, 1.0, out=g_out)

    # store back to original g (float16)
    g[:] = g_out.astype(np.float16, copy=False)


def refine_alpha(alpha: np.ndarray, feather: float, clamp_low: float, clamp_high: float, hard_alpha: bool) -> np.ndarray:
    """
    alpha: float32 [H,W]
    Use PIL GaussianBlur to avoid huge numpy padding arrays.
    """
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)

    if feather > 0:
        # Convert to 8-bit for blur (fast & low memory), then back to float
        a8 = (a * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(a8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(feather)))
        a = (np.asarray(img, dtype=np.float32) / 255.0)

    a[a < clamp_low] = 0.0
    a[a > clamp_high] = 1.0

    if hard_alpha:
        # smoothstep hardening (keeps thin lines better than harsh threshold)
        a = np.clip(a, 0.0, 1.0)
        a = (a * a) * (3 - 2 * a)
        a[a < clamp_low] = 0.0
        a[a > clamp_high] = 1.0

    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def process_image(im: Image.Image, p: Params) -> Image.Image:
    im = im.convert("RGB")

    # Safety: if someone uploads huge images, keep it bounded
    # (won't affect 4K use-cases; only protects memory blow-ups)
    max_side = 5000
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)

    rgb_u8 = np.asarray(im, dtype=np.uint8)  # [H,W,3] uint8 (low memory)
    lut = srgb_to_linear_lut(dtype=np.float16)

    # 1) Linearize per-channel into float16 (much less RAM than float32 RGB)
    r = lut[rgb_u8[..., 0]]
    g = lut[rgb_u8[..., 1]]
    b = lut[rgb_u8[..., 2]]

    # 2) Aggressive chroma alpha (float32, single channel)
    alpha = compute_chroma_alpha_linear(r, g, b, p.green_tolerance)

    # 3) Edge band mask (PIL morphology, low peak RAM)
    edge_mask = edge_band_from_alpha(alpha, p.edge_band_px)

    # 4) Edge-only spill suppression (in-place on g)
    spill_suppress_edge_only_inplace(r, g, b, edge_mask, p.spill_strength)

    # 5) Alpha refine
    alpha2 = refine_alpha(alpha, p.feather, p.alpha_clamp_low, p.alpha_clamp_high, p.hard_alpha)

    # Compose RGBA (convert linear->sRGB)
    # Use float32 for conversion, then to uint8
    r32 = r.astype(np.float32, copy=False)
    g32 = g.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)

    rgb_lin = np.stack([r32, g32, b32], axis=-1)  # small extra but needed for output
    rgb_out = linear_to_srgb(rgb_lin)
    rgb_u8_out = np.clip(rgb_out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    a_u8 = np.clip(alpha2 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb_u8_out, a_u8])
    return Image.fromarray(rgba, mode="RGBA")


# ----------------------------
# FastAPI routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    green_tolerance: int = Form(40),
    edge_band_px: int = Form(4),
    spill_strength: int = Form(65),
    feather: float = Form(0.6),
    alpha_clamp_low: float = Form(0.03),
    alpha_clamp_high: float = Form(0.98),
    hard_alpha: Optional[str] = Form(None),
):
    data = await file.read()
    im = Image.open(io.BytesIO(data))

    p = Params(
        green_tolerance=int(np.clip(green_tolerance, 0, 255)),
        edge_band_px=int(np.clip(edge_band_px, 1, 12)),
        spill_strength=int(np.clip(spill_strength, 0, 100)),
        feather=float(np.clip(feather, 0.0, 2.0)),
        alpha_clamp_low=float(np.clip(alpha_clamp_low, 0.0, 0.2)),
        alpha_clamp_high=float(np.clip(alpha_clamp_high, 0.8, 1.0)),
        hard_alpha=(hard_alpha is not None),
    )

    out = process_image(im, p)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    png = buf.getvalue()

    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=cleaned.png"},
    )
