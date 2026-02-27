import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI(title="Chroma Key Cleaner (512MB Safe)")
templates = Jinja2Templates(directory="templates")


@dataclass
class Params:
    green_tolerance: int = 40      # 0..255 (higher = more aggressive key)
    spill_strength: int = 65       # 0..100
    feather: float = 0.6           # 0..2
    alpha_clamp_low: float = 0.03  # 0..0.2
    alpha_clamp_high: float = 0.98 # 0.8..1.0
    hard_alpha: bool = True
    max_side: int = 2600           # TEST MODE cap (change later to 4200 if stable)


def compute_alpha_u8(rgb_u8: np.ndarray, tol: int) -> np.ndarray:
    """
    Compute chroma alpha as uint8 (0..255) with minimal memory.
    Works well for neon-green background.
    """
    # Use int16 for dominance to avoid uint8 underflow
    r = rgb_u8[..., 0].astype(np.int16, copy=False)
    g = rgb_u8[..., 1].astype(np.int16, copy=False)
    b = rgb_u8[..., 2].astype(np.int16, copy=False)

    maxrb = np.maximum(r, b)
    dom = g - maxrb  # green dominance

    # Convert tolerance to thresholds (empirical, stable for AI chroma bgs)
    # higher tol => remove more
    # We also require green to be reasonably high to avoid killing dark lines.
    g_min = 40 + int(tol * 0.35)          # green must be bright-ish to be background
    dom_cut = 18 - int(tol * 0.06)        # dominance cutoff
    dom_soft = 18 + int(tol * 0.04)       # softness range

    # Background likelihood based on dom and green brightness
    # score in roughly [0..255]
    dom_clamped = np.clip(dom - dom_cut, 0, dom_soft).astype(np.float32)
    dom_norm = dom_clamped / max(1.0, float(dom_soft))  # 0..1
    g_gate = np.clip((g - g_min) / 80.0, 0.0, 1.0)      # 0..1 gate
    bg = dom_norm * g_gate                               # 0..1

    # alpha = 1 - smoothstep(bg)
    # smoothstep for nicer edges without heavy blur
    s = bg * bg * (3.0 - 2.0 * bg)
    alpha = 1.0 - s
    a_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return a_u8


def spill_suppress_edge_only(rgb_u8: np.ndarray, alpha_u8: np.ndarray, strength: int) -> None:
    """
    In-place edge-only spill suppression on uint8 RGB.
    Edge zone defined cheaply: 0 < alpha < 255.
    Only adjust G where green dominates and alpha is in edge zone.
    """
    if strength <= 0:
        return

    a = alpha_u8
    edge = (a > 0) & (a < 255)

    r = rgb_u8[..., 0].astype(np.int16, copy=False)
    g = rgb_u8[..., 1].astype(np.int16, copy=False)
    b = rgb_u8[..., 2].astype(np.int16, copy=False)

    maxrb = np.maximum(r, b)
    green_dom = g > (maxrb + 2)  # +2 to avoid touching neutral pixels

    m = edge & green_dom
    if not np.any(m):
        return

    # target green = roughly average of R and B (pull green back)
    target = ((r + b) // 2)

    # blend: g_new = (1-k)*g + k*target
    k = strength / 100.0
    g_new = (g.astype(np.float32) * (1.0 - k) + target.astype(np.float32) * k)
    g_new_u8 = np.clip(g_new + 0.5, 0, 255).astype(np.uint8)

    # write back only masked pixels
    g_chan = rgb_u8[..., 1]
    g_chan[m] = g_new_u8[m]


def refine_alpha_u8(alpha_u8: np.ndarray, feather: float, clamp_low: float, clamp_high: float, hard_alpha: bool) -> np.ndarray:
    """
    Alpha refinement in uint8 to keep memory low.
    """
    a = alpha_u8

    if feather > 0:
        img = Image.fromarray(a, mode="L").filter(ImageFilter.GaussianBlur(radius=float(feather)))
        a = np.asarray(img, dtype=np.uint8)

    low = int(np.clip(clamp_low, 0.0, 0.2) * 255.0 + 0.5)
    high = int(np.clip(clamp_high, 0.8, 1.0) * 255.0 + 0.5)

    # clamp noise
    a = a.copy()
    a[a < low] = 0
    a[a > high] = 255

    if hard_alpha:
        # mild hardening: push mids slightly without killing thin lines
        # use a simple LUT curve: smoothstep-like on 0..255
        x = (a.astype(np.float32) / 255.0)
        y = x * x * (3.0 - 2.0 * x)  # smoothstep
        a2 = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        a2[a2 < low] = 0
        a2[a2 > high] = 255
        a = a2

    return a


def process_image(im: Image.Image, p: Params) -> Image.Image:
    im = im.convert("RGB")

    # TEST MODE resize cap
    if max(im.size) > p.max_side:
        im.thumbnail((p.max_side, p.max_side), Image.LANCZOS)

    rgb_u8 = np.array(im, dtype=np.uint8, copy=True)  # ensure writable

    alpha_u8 = compute_alpha_u8(rgb_u8, p.green_tolerance)
    spill_suppress_edge_only(rgb_u8, alpha_u8, p.spill_strength)
    alpha_u8 = refine_alpha_u8(alpha_u8, p.feather, p.alpha_clamp_low, p.alpha_clamp_high, p.hard_alpha)

    rgba = np.dstack([rgb_u8, alpha_u8])
    return Image.fromarray(rgba, mode="RGBA")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    green_tolerance: int = Form(40),
    spill_strength: int = Form(65),
    feather: float = Form(0.6),
    alpha_clamp_low: float = Form(0.03),
    alpha_clamp_high: float = Form(0.98),
    hard_alpha: Optional[str] = Form(None),
):
    # IMPORTANT: do not read into bytes -> avoids duplicate memory buffer
    file.file.seek(0)
    im = Image.open(file.file)

    p = Params(
        green_tolerance=int(np.clip(green_tolerance, 0, 255)),
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
