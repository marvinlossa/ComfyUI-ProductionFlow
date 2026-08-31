"""Common still / frame filters on ComfyUI IMAGE batches (N, H, W, C) in [0, 1]."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .motion_blur_film_grain import apply_film_grain

ProgressCallback = Optional[Callable[[int], None]]

_LUMA = (0.2126, 0.7152, 0.0722)


def _vignette_mask(height: int, width: int, device, dtype, strength: float) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).unsqueeze(1)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).unsqueeze(0)
    radius = torch.sqrt(xx * xx + yy * yy)
    t = ((radius - 0.35) / 0.80).clamp(0.0, 1.0)
    t = t * t
    return 1.0 - float(strength) * t


def apply_image_filters(
    images: torch.Tensor,
    saturation: float = 1.0,
    contrast: float = 1.0,
    warmth: float = 0.0,
    vignette: float = 0.0,
    grain: float = 0.0,
    seed: int = 0,
    progress_callback: ProgressCallback = None,
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE tensor NHWC, got shape {tuple(images.shape)}")

    n = images.shape[0]
    sat = float(saturation)
    con = float(contrast)
    warm = float(warmth)
    vig = float(vignette)
    grain_s = float(grain)

    do_sat = abs(sat - 1.0) > 1e-6
    do_con = abs(con - 1.0) > 1e-6
    do_warm = abs(warm) > 1e-6
    do_vig = vig > 1e-6
    do_grain = grain_s > 0.0

    if not (do_sat or do_con or do_warm or do_vig or do_grain):
        if progress_callback is not None and n > 0:
            progress_callback(n)
        return images

    out = images
    if do_sat or do_con or do_warm or do_vig:
        out = images.clone()
        rgb = out[..., :3]
        if do_con:
            rgb.sub_(0.5).mul_(con).add_(0.5)
        if do_sat:
            weights = torch.tensor(_LUMA, device=rgb.device, dtype=rgb.dtype)
            luma = (rgb * weights).sum(dim=-1, keepdim=True)
            rgb.lerp_(luma, 1.0 - sat)
        if do_warm:
            rgb[..., 0].add_(warm * 0.10)
            rgb[..., 1].add_(warm * 0.03)
            rgb[..., 2].sub_(warm * 0.08)
        if do_vig:
            h, w = rgb.shape[1], rgb.shape[2]
            mask = _vignette_mask(h, w, rgb.device, rgb.dtype, vig)
            rgb.mul_(mask.view(1, h, w, 1))
        rgb.clamp_(0.0, 1.0)

    if do_grain:
        out = apply_film_grain(
            out,
            grain_s,
            seed=seed,
            inplace=(out is not images),
            progress_callback=progress_callback,
        )
    elif progress_callback is not None and n > 0:
        progress_callback(n)
    return out
