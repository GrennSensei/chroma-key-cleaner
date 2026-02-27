import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI(title="Chroma Key Cleaner (Ultra Low-Memory)")
templates = Jinja2Templates(directory="templates")


# ----------------------------
# LUTs for sRGB <-> linear
# ----------------------------
_SRGB_TO_LINEAR_LUT_F16 = None
_LINEAR_TO_SRGB_LUT_U8 = None

def srgb_to_linear_lut_f16() -> np.ndarray:
    global _SRGB_TO_LINEAR_LUT_F16
    if _SRGB_TO_LINEAR_LUT_F16 is None:
        x = np.linspace(0, 1, 256, dtype=np.float32)
        a = 0.055
        lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
        _SRGB_TO_LINEAR_LUT_F16 = lin.astype(np.float16)
    return _SRGB_TO_LINEAR_LUT_F16

def linear_to_srgb_u8_channel(x_lin: np.ndarray) -> np.ndarray:
    """
    Convert a single linear float channel [H,W] -> sRGB uint8 using formula (no 3D stacks).
    x_lin should be float16/float32 in [0,1].
    """
    x = x_lin.astype(np.float32, copy=False)
    a = 0.055
    srgb = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------
# Edge band detection (RAM friendly)
# ----------------------------
def edge_band_from_alpha_fast(alpha: np.ndarray, band_px: int) -> np.ndarray:
    """
    alpha: float32 [H,W] in [0,1]
    Returns bool edge mask where alpha changes rapidly, expanded to band_px.
    No morphology, no padding-heavy ops.
    """
    a = alpha.astype(np.float32, copy=False)

    # simple gradients (no pad): use differences
    gx = np.abs(a[:, 1:] - a[:, :-1])
    gy = np.abs(a[1:, :] - a[:-1, :])

    # bring to [H,W] by padding minimal (tiny)
    g = np.zeros_like(a, dtype=np.float32)
    g[:, 1:] += gx
    g[1:, :] += gy

    # Edge where gradient is noticeable
    edge = g > 0.03

    if band_px <= 1:
        return edge

    # Expand edge by blurring the boolean mask cheaply:
    # Convert to L image, box blur, then threshold.
    edge_u8 = (edge.astype(np.uint8) * 255)
    img = Image.fromarray(edge_u8, mode="L").filter(ImageFilter.BoxBlur(radius=int(band_px)))
    expanded = (np.asarray(img, dtype=np.uint8) > 0)
    return expanded


# ----------------------------
# Core pipeline
# ----------------------------
@dataclass
class Params:
    green_tolerance: int = 40
    edge_band_px: int = 4
    spill_strength: int = 65
    feather: float = 0.6
    alpha_clamp_low: float = 0.03
    alpha_clamp_high: float = 0.98
    hard_alpha: bool = True
    max_side: int = 4200   # HARD cap for 512MB safety (4K-ish)


def compute_chroma_alpha_linear(r: np.ndarray, g: np.ndarray, b: np.ndarray, tol: int) -> np.ndarray:
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
    if strength <= 0:
        return

    k = strength / 100.0

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

    # Update only masked pixels
    g_out = g32.copy()
    g_out[m] = g_new[m]
    np.clip(g_out, 0.0, 1.0, out=g_out)
    g[:] = g_out.astype(np.float16, copy=False)


def refine_alpha(alpha: np.ndarray, feather: float, clamp_low: float, clamp_high: float, hard_alpha: bool) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)

    if feather > 0:
        a8 = (a * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(a8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(feather)))
        a = (np.asarray(img, dtype=np.float32) / 255.0)

    a[a < clamp_low] = 0.0
    a[a > clamp_high] = 1.0

    if hard_alpha:
        a = np.clip(a, 0.0, 1.0)
        a = (a * a) * (3 - 2 * a)
        a[a < clamp_low] = 0.0
        a[a > clamp_high] = 1.0

    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def process_image(im: Image.Image, p: Params) -> Image.Image:
    im = im.convert("RGB")

    # Hard cap for Render 512MB safety
    if max(im.size) > p.max_side:
        im.thumbnail((p.max_side, p.max_side), Image.LANCZOS)

    rgb_u8 = np.asarray(im, dtype=np.uint8)
    lut = srgb_to_linear_lut_f16()

    # Linearize channels as float16 (low RAM)
    r = lut[rgb_u8[..., 0]]
    g = lut[rgb_u8[..., 1]]
    b = lut[rgb_u8[..., 2]]

    alpha = compute_chroma_alpha_linear(r, g, b, p.green_tolerance)

    # RAM-friendly edge band
    edge_mask = edge_band_from_alpha_fast(alpha, p.edge_band_px)

    spill_suppress_edge_only_inplace(r, g, b, edge_mask, p.spill_strength)

    alpha2 = refine_alpha(alpha, p.feather, p.alpha_clamp_low, p.alpha_clamp_high, p.hard_alpha)

    # Convert to sRGB uint8 channel-by-channel (no 3D float stacks)
    r8 = linear_to_srgb_u8_channel(r)
    g8 = linear_to_srgb_u8_channel(g)
    b8 = linear_to_srgb_u8_channel(b)
    a8 = np.clip(alpha2 * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rgba = np.dstack([r8, g8, b8, a8])
    return Image.fromarray(rgba, mode="RGBA")


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
