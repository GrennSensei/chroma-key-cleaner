import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI(title="Chroma Key Cleaner (Memory Safe + Decontaminate)")
templates = Jinja2Templates(directory="templates")


# Fixed neon green background (approx #00FF00)
BG_R = 0
BG_G = 255
BG_B = 0


@dataclass
class Params:
    green_tolerance: int = 40       # 0..255
    spill_strength: int = 65        # 0..100  (extra clamp after decontaminate)
    feather: float = 0.6            # 0..2
    alpha_clamp_low: float = 0.03   # 0..0.2
    alpha_clamp_high: float = 0.98  # 0.8..1
    hard_alpha: bool = True
    decontaminate: bool = True
    ultra_clean: bool = False       # 1px inward matte
    edge_expand_px: int = 2         # expand edge band to catch halo
    max_side: int = 3000            # you can raise later (e.g. 4200) if stable


def compute_alpha_u8(rgb_u8: np.ndarray, tol: int) -> np.ndarray:
    """
    Memory-safe chroma alpha as uint8 (0..255).
    Targets neon green background.
    """
    r = rgb_u8[..., 0].astype(np.int16, copy=False)
    g = rgb_u8[..., 1].astype(np.int16, copy=False)
    b = rgb_u8[..., 2].astype(np.int16, copy=False)

    maxrb = np.maximum(r, b)
    dom = g - maxrb  # green dominance

    # Tolerance shaping
    t = int(np.clip(tol, 0, 255))
    g_min = 35 + int(t * 0.35)          # require green brightness
    dom_cut = 16 - int(t * 0.05)        # dominance cutoff
    dom_soft = 20 + int(t * 0.05)       # softness

    dom_clamped = np.clip(dom - dom_cut, 0, dom_soft).astype(np.float32)
    dom_norm = dom_clamped / max(1.0, float(dom_soft))  # 0..1
    g_gate = np.clip((g - g_min) / 90.0, 0.0, 1.0)      # 0..1
    bg = dom_norm * g_gate                               # 0..1

    # smoothstep
    s = bg * bg * (3.0 - 2.0 * bg)
    alpha = 1.0 - s
    return np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)


def edge_mask_from_alpha(alpha_u8: np.ndarray, expand_px: int) -> np.ndarray:
    """
    Edge zone where alpha is not fully 0 or 255, expanded to catch halo pixels.
    Uses PIL MaxFilter (low memory).
    """
    edge = (alpha_u8 > 0) & (alpha_u8 < 255)
    if expand_px <= 0:
        return edge

    img = Image.fromarray((edge.astype(np.uint8) * 255), mode="L")
    # Expand edge region
    size = 2 * expand_px + 1
    img2 = img.filter(ImageFilter.MaxFilter(size=size))
    return (np.asarray(img2, dtype=np.uint8) > 0)


def refine_alpha_u8(alpha_u8: np.ndarray, feather: float, clamp_low: float, clamp_high: float, hard_alpha: bool) -> np.ndarray:
    a = alpha_u8

    if feather > 0:
        img = Image.fromarray(a, mode="L").filter(ImageFilter.GaussianBlur(radius=float(feather)))
        a = np.asarray(img, dtype=np.uint8)

    low = int(np.clip(clamp_low, 0.0, 0.2) * 255.0 + 0.5)
    high = int(np.clip(clamp_high, 0.8, 1.0) * 255.0 + 0.5)

    a = a.copy()
    a[a < low] = 0
    a[a > high] = 255

    if hard_alpha:
        x = (a.astype(np.float32) / 255.0)
        y = x * x * (3.0 - 2.0 * x)  # smoothstep
        a2 = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        a2[a2 < low] = 0
        a2[a2 > high] = 255
        a = a2

    return a


def ultra_clean_inward(alpha_u8: np.ndarray, px: int = 1) -> np.ndarray:
    """
    1px inward matte (erode alpha>0 mask), then apply to alpha.
    Helps remove last 1px halo for print-ready output.
    """
    if px <= 0:
        return alpha_u8

    fg = alpha_u8 > 0
    img = Image.fromarray((fg.astype(np.uint8) * 255), mode="L")
    size = 2 * px + 1
    # MinFilter erodes
    img2 = img.filter(ImageFilter.MinFilter(size=size))
    fg2 = (np.asarray(img2, dtype=np.uint8) > 0)

    out = alpha_u8.copy()
    out[~fg2] = 0
    return out


def decontaminate_edges_inplace(rgb_u8: np.ndarray, alpha_u8: np.ndarray, edge_mask: np.ndarray):
    """
    Edge-only foreground color recovery:
    F ≈ (C - (1-a)*Bg) / a
    where Bg = (0,255,0).
    Runs ONLY on edge_mask to preserve interiors.
    Memory-safe: uses float32 only on masked pixels.
    """
    m = edge_mask
    if not np.any(m):
        return

    a = alpha_u8.astype(np.float32, copy=False) / 255.0
    a_m = a[m]
    # Avoid division blow-up
    eps = 1.0 / 255.0
    denom = np.maximum(a_m, eps)

    # Read channels as float32 for masked pixels only
    r = rgb_u8[..., 0].astype(np.float32, copy=False)
    g = rgb_u8[..., 1].astype(np.float32, copy=False)
    b = rgb_u8[..., 2].astype(np.float32, copy=False)

    r_m = r[m]
    g_m = g[m]
    b_m = b[m]

    one_minus = (1.0 - a_m)

    # Subtract background contribution and unmix
    r_f = (r_m - one_minus * BG_R) / denom
    g_f = (g_m - one_minus * BG_G) / denom
    b_f = (b_m - one_minus * BG_B) / denom

    # Clip back to valid range
    r_f = np.clip(r_f, 0.0, 255.0)
    g_f = np.clip(g_f, 0.0, 255.0)
    b_f = np.clip(b_f, 0.0, 255.0)

    # Write back
    rgb_u8[..., 0][m] = (r_f + 0.5).astype(np.uint8)
    rgb_u8[..., 1][m] = (g_f + 0.5).astype(np.uint8)
    rgb_u8[..., 2][m] = (b_f + 0.5).astype(np.uint8)


def clamp_green_spill_inplace(rgb_u8: np.ndarray, edge_mask: np.ndarray, strength: int):
    """
    Extra safety: after decontamination, clamp residual green dominance in edge zone.
    This is general (not red-specific).
    """
    if strength <= 0:
        return

    k = strength / 100.0
    m = edge_mask
    if not np.any(m):
        return

    r = rgb_u8[..., 0].astype(np.int16, copy=False)
    g = rgb_u8[..., 1].astype(np.int16, copy=False)
    b = rgb_u8[..., 2].astype(np.int16, copy=False)

    maxrb = np.maximum(r, b)
    green_dom = g > (maxrb + 1)  # tiny margin

    mm = m & green_dom
    if not np.any(mm):
        return

    target = ((r + b) // 2).astype(np.float32)
    g_f = g.astype(np.float32)

    g_new = (1.0 - k) * g_f + k * target
    g_new_u8 = np.clip(g_new + 0.5, 0, 255).astype(np.uint8)

    rgb_u8[..., 1][mm] = g_new_u8[mm]


def process_image(im: Image.Image, p: Params) -> Image.Image:
    im = im.convert("RGB")

    # Safety cap for Render free tier
    if max(im.size) > p.max_side:
        im.thumbnail((p.max_side, p.max_side), Image.LANCZOS)

    # Ensure writable array
    rgb_u8 = np.array(im, dtype=np.uint8, copy=True)

    # 1) Alpha
    alpha_u8 = compute_alpha_u8(rgb_u8, p.green_tolerance)

    # 2) Refine alpha early (helps edge band stability)
    alpha_u8 = refine_alpha_u8(alpha_u8, p.feather, p.alpha_clamp_low, p.alpha_clamp_high, p.hard_alpha)

    # 3) Expanded edge band
    edge_mask = edge_mask_from_alpha(alpha_u8, p.edge_expand_px)

    # 4) Decontaminate edges (unmix from neon green bg)
    if p.decontaminate:
        decontaminate_edges_inplace(rgb_u8, alpha_u8, edge_mask)

    # 5) Extra green clamp in edge zone
    clamp_green_spill_inplace(rgb_u8, edge_mask, p.spill_strength)

    # 6) Optional ultra clean 1px inward matte
    if p.ultra_clean:
        alpha_u8 = ultra_clean_inward(alpha_u8, px=1)

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
    decontaminate: Optional[str] = Form(None),
    ultra_clean: Optional[str] = Form(None),
):
    # stream open, avoid read() duplicating memory
    file.file.seek(0)
    im = Image.open(file.file)

    p = Params(
        green_tolerance=int(np.clip(green_tolerance, 0, 255)),
        spill_strength=int(np.clip(spill_strength, 0, 100)),
        feather=float(np.clip(feather, 0.0, 2.0)),
        alpha_clamp_low=float(np.clip(alpha_clamp_low, 0.0, 0.2)),
        alpha_clamp_high=float(np.clip(alpha_clamp_high, 0.8, 1.0)),
        hard_alpha=(hard_alpha is not None),
        decontaminate=(decontaminate is not None),
        ultra_clean=(ultra_clean is not None),
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
