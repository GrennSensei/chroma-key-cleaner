import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI(title="Chroma Key Cleaner")
templates = Jinja2Templates(directory="templates")


# ----------------------------
# Color space helpers (sRGB <-> Linear)
# ----------------------------
def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    # u: float in [0,1]
    a = 0.055
    return np.where(u <= 0.04045, u / 12.92, ((u + a) / (1 + a)) ** 2.4)

def linear_to_srgb(u: np.ndarray) -> np.ndarray:
    # u: float in [0,1]
    a = 0.055
    return np.where(u <= 0.0031308, 12.92 * u, (1 + a) * (u ** (1 / 2.4)) - a)


# ----------------------------
# Small image ops (no OpenCV)
# ----------------------------
def blur2d(img: np.ndarray, radius: float) -> np.ndarray:
    """
    Lightweight separable gaussian-ish blur using 1D kernel.
    img: 2D float array [H,W]
    radius: 0..2 typically
    """
    if radius <= 0:
        return img

    sigma = max(0.1, radius)
    # kernel size: ~ 6*sigma
    k = int(max(3, math.ceil(sigma * 6)))
    if k % 2 == 0:
        k += 1
    half = k // 2

    x = np.arange(-half, half + 1, dtype=np.float32)
    kern = np.exp(-(x * x) / (2 * sigma * sigma))
    kern /= np.sum(kern)

    # pad
    padded = np.pad(img, ((0, 0), (half, half)), mode="edge")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[1]):
        tmp[:, i] = np.sum(padded[:, i:i + k] * kern[None, :], axis=1)

    padded2 = np.pad(tmp, ((half, half), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(img.shape[0]):
        out[j, :] = np.sum(padded2[j:j + k, :] * kern[:, None], axis=0)

    return out


def dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    # mask: bool [H,W]
    m = mask.copy()
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neigh = (
            p[0:-2, 0:-2] | p[0:-2, 1:-1] | p[0:-2, 2:] |
            p[1:-1, 0:-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
            p[2:, 0:-2] | p[2:, 1:-1] | p[2:, 2:]
        )
        m = neigh
    return m


def erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.copy()
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neigh = (
            p[0:-2, 0:-2] & p[0:-2, 1:-1] & p[0:-2, 2:] &
            p[1:-1, 0:-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
            p[2:, 0:-2] & p[2:, 1:-1] & p[2:, 2:]
        )
        m = neigh
    return m


def edge_band_from_alpha(alpha: np.ndarray, band_px: int) -> np.ndarray:
    """
    alpha: float [H,W] in [0,1]
    returns bool mask of pixels near the edge (transition zone)
    """
    fg = alpha > 0.5
    # Edge = dilate(fg) - erode(fg)
    d = dilate(fg, 1)
    e = erode(fg, 1)
    edge = d & (~e)

    # Expand edge to band
    band = edge.copy()
    if band_px > 1:
        band = dilate(band, band_px - 1)
    return band


# ----------------------------
# Core pipeline
# ----------------------------
@dataclass
class Params:
    green_tolerance: int = 40            # 0..255
    edge_band_px: int = 4                # 1..12
    spill_strength: int = 65             # 0..100
    feather: float = 0.6                 # 0..2
    alpha_clamp_low: float = 0.03        # 0..0.2
    alpha_clamp_high: float = 0.98       # 0.8..1
    hard_alpha: bool = True


def compute_chroma_alpha_linear(rgb_lin: np.ndarray, tol: int) -> np.ndarray:
    """
    Aggressive chroma alpha in linear space.
    rgb_lin: float [H,W,3] linear 0..1
    tol: 0..255 in sRGB-ish scale (we map it to 0..1 threshold)
    Returns alpha float [H,W] where 1=foreground.
    """
    r = rgb_lin[..., 0]
    g = rgb_lin[..., 1]
    b = rgb_lin[..., 2]

    # "How green" metric: green dominates over red/blue
    # We want to remove background where green is strongly dominant.
    # Convert tol to a dominance threshold in 0..1
    # Higher tol -> easier to treat pixels as background (more aggressive)
    t = tol / 255.0

    # dominance: g - max(r,b)
    dom = g - np.maximum(r, b)   # -1..1 approximately
    # background likelihood: if dom is high AND g is high
    # We also require g to be moderately bright to avoid killing dark strokes.
    bg_score = dom * (g + 0.15)

    # Map score to alpha with a soft step:
    # choose cutoff based on t
    cutoff = 0.08 - (t * 0.06)   # higher t => lower cutoff => more bg removed
    softness = 0.03 + (t * 0.02)

    # alpha = 1 - smoothstep(cutoff, cutoff+softness, bg_score)
    x = (bg_score - cutoff) / max(1e-6, softness)
    x = np.clip(x, 0.0, 1.0)
    # smoothstep
    s = x * x * (3 - 2 * x)
    alpha = 1.0 - s
    return np.clip(alpha, 0.0, 1.0)


def spill_suppress_edge_only(rgb_lin: np.ndarray, edge_mask: np.ndarray, strength: int) -> np.ndarray:
    """
    Edge-only green spill suppression in linear space.
    We reduce green dominance by pulling G towards min(R,B) within edge zone.
    """
    if strength <= 0:
        return rgb_lin

    k = strength / 100.0
    out = rgb_lin.copy()

    r = out[..., 0]
    g = out[..., 1]
    b = out[..., 2]

    target_g = (np.minimum(r, b) * 0.8) + ((r + b) * 0.5 * 0.2)  # stable target
    # Only where green is dominant:
    green_dom = g > (np.maximum(r, b) + 1e-6)

    m = edge_mask & green_dom
    if not np.any(m):
        return out

    # Blend g towards target
    g_new = (1.0 - k) * g + k * target_g
    out[..., 1] = np.where(m, g_new, g)

    return np.clip(out, 0.0, 1.0)


def refine_alpha(alpha: np.ndarray, feather: float, clamp_low: float, clamp_high: float, hard_alpha: bool) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)

    # Optional feather (small blur)
    if feather > 0:
        a = blur2d(a.astype(np.float32), radius=feather)

    # Clamp to fight noise / halos
    a = np.where(a < clamp_low, 0.0, a)
    a = np.where(a > clamp_high, 1.0, a)

    # Print-ready hardening (very mild)
    if hard_alpha:
        # push mid values towards 0/1 slightly without destroying thin lines
        # curve: a' = a^gamma with gamma<1 boosts, gamma>1 reduces. We'll do S-curve-ish.
        a = np.clip(a, 0.0, 1.0)
        a = (a * a) * (3 - 2 * a)  # smoothstep
        # keep clamps after curve
        a = np.where(a < clamp_low, 0.0, a)
        a = np.where(a > clamp_high, 1.0, a)

    return np.clip(a, 0.0, 1.0)


def process_image(im: Image.Image, p: Params) -> Image.Image:
    # Ensure RGB
    im = im.convert("RGB")
    rgb = np.asarray(im).astype(np.float32) / 255.0  # sRGB 0..1

    # 1) Linearize
    rgb_lin = srgb_to_linear(rgb)

    # 2) Aggressive chroma key alpha (linear)
    alpha = compute_chroma_alpha_linear(rgb_lin, p.green_tolerance)

    # 3) Edge band mask from alpha
    edge_mask = edge_band_from_alpha(alpha, p.edge_band_px)

    # 4) Edge-only spill suppression (color correction, NOT alpha kill)
    rgb_lin2 = spill_suppress_edge_only(rgb_lin, edge_mask, p.spill_strength)

    # 5) Alpha refine (feather + clamp + print hardening)
    alpha2 = refine_alpha(alpha, p.feather, p.alpha_clamp_low, p.alpha_clamp_high, p.hard_alpha)

    # Compose RGBA in sRGB
    rgb_out = linear_to_srgb(rgb_lin2)
    rgb_u8 = np.clip(rgb_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    a_u8 = np.clip(alpha2 * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rgba = np.dstack([rgb_u8, a_u8])
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
    hard_alpha: str | None = Form(None),
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

    return Response(content=png, media_type="image/png", headers={
        "Content-Disposition": "inline; filename=cleaned.png"
    })
