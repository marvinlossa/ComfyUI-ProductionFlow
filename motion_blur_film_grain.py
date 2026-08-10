"""Temporal motion blur then film grain on image batches.

Mirrors the filter order and defaults from MotionBlurFilmGrain
(tmix-style temporal blend, then noise-style grain). Operates on
ComfyUI IMAGE tensors (B, H, W, C) in float [0, 1] and returns images —
no video encode / MP4 export.

Processing is done in small frame chunks so peak RAM stays near one output
buffer and ComfyUI ProgressBar can advance frame-by-frame.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

# Frames processed per step (blur + grain). Small enough for smooth progress
# and low peak RAM; large enough to keep Python-loop overhead low.
_PROCESS_CHUNK_FRAMES = 4

ProgressCallback = Optional[Callable[[int], None]]


def _normalize_window(blur_window: int) -> int:
    window = int(blur_window)
    if window < 1:
        return 1
    if window % 2 == 0:
        window += 1
    return window


def build_tmix_weight_list(window: int, strength: float) -> list[float]:
    """Same weight layout as the CLI: center=1, neighbors=strength, normalized."""
    if window <= 1 or strength <= 0:
        return [1.0]
    center = window // 2
    values = [1.0 if i == center else float(strength) for i in range(window)]
    total = sum(values)
    return [v / total for v in values]


def build_tmix_weights(window: int, strength: float, device, dtype) -> torch.Tensor:
    """Tensor form of build_tmix_weight_list (kept for callers/tests)."""
    values = build_tmix_weight_list(window, strength)
    return torch.tensor(values, device=device, dtype=dtype)


def _report(progress_callback: ProgressCallback, amount: int) -> None:
    if progress_callback is not None and amount > 0:
        progress_callback(amount)


def _blur_chunk(
    out_chunk: torch.Tensor,
    images: torch.Tensor,
    start: int,
    end: int,
    weights: list[float],
    center: int,
) -> None:
    """Write temporal mix for frames [start, end) into out_chunk."""
    n = images.shape[0]
    length = end - start
    out_chunk.zero_()

    # Reusable index buffer on the same device as images (usually CPU for IMAGE).
    base = torch.arange(start, end, device=images.device, dtype=torch.long)

    for i, weight in enumerate(weights):
        w = float(weight)
        if w == 0.0:
            continue
        shift = i - center
        if shift == 0:
            out_chunk.add_(images[start:end], alpha=w)
            continue
        src = (base + shift).clamp_(0, n - 1)
        out_chunk.add_(images[src], alpha=w)


def _grain_chunk(
    chunk: torch.Tensor,
    amplitude: float,
    generator: torch.Generator,
) -> None:
    """Add uniform film grain in-place to a small frame chunk."""
    noise = torch.rand(
        chunk.shape,
        generator=generator,
        device=chunk.device,
        dtype=chunk.dtype,
    )
    noise.mul_(2.0).sub_(1.0).mul_(amplitude)
    chunk.add_(noise).clamp_(0.0, 1.0)
    del noise


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        return generator
    except Exception:
        generator = torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFF)
        return generator


@torch.inference_mode()
def temporal_motion_blur(
    images: torch.Tensor,
    blur_window: int,
    blur_strength: float,
    progress_callback: ProgressCallback = None,
    chunk_frames: int = _PROCESS_CHUNK_FRAMES,
) -> torch.Tensor:
    """Temporal mix of neighboring frames (ffmpeg tmix-style).

    images: (N, H, W, C) float tensor in [0, 1].
    Processes output frames in chunks; optional progress_callback(n_frames).
    """
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE tensor NHWC, got shape {tuple(images.shape)}")

    n = images.shape[0]
    if n == 0:
        return images

    window = _normalize_window(blur_window)
    strength = float(blur_strength)
    if window == 1 or strength <= 0 or n == 1:
        _report(progress_callback, n)
        return images

    weights = build_tmix_weight_list(window, strength)
    center = window // 2
    chunk = max(1, int(chunk_frames))

    out = torch.empty(images.shape, dtype=images.dtype, device=images.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        _blur_chunk(out[start:end], images, start, end, weights, center)
        _report(progress_callback, end - start)
    return out


@torch.inference_mode()
def apply_film_grain(
    images: torch.Tensor,
    grain_strength: float,
    seed: int = 0,
    chunk_frames: int = _PROCESS_CHUNK_FRAMES,
    inplace: bool = False,
    progress_callback: ProgressCallback = None,
) -> torch.Tensor:
    """Uniform temporal film grain roughly matching ffmpeg noise=alls=S:allf=t+u.

    ffmpeg noise strength is on an 8-bit scale (0–100 typical). We map that to
    float [0, 1] by dividing by 255. Fresh noise each frame (temporal).
    """
    strength = float(grain_strength)
    n = images.shape[0] if images.ndim >= 1 else 0
    if strength <= 0 or images.numel() == 0:
        _report(progress_callback, n)
        return images

    out = images if inplace else images.clone()
    amplitude = strength / 255.0
    generator = _make_generator(out.device, seed)
    chunk = max(1, int(chunk_frames))

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        _grain_chunk(out[start:end], amplitude, generator)
        _report(progress_callback, end - start)

    return out


@torch.inference_mode()
def apply_motion_blur_film_grain(
    images: torch.Tensor,
    blur_window: int = 3,
    blur_strength: float = 0.35,
    grain_strength: float = 6.0,
    seed: int = 0,
    enable_blur: bool = True,
    enable_grain: bool = True,
    progress_callback: ProgressCallback = None,
    chunk_frames: int = _PROCESS_CHUNK_FRAMES,
) -> torch.Tensor:
    """Apply motion blur first, then grain — same order as the CLI filter chain.

    Progress is reported in frame units. When both effects run, total progress
    spans 2 * N (blur pass then grain pass) so the bar advances through both.
    When only one effect runs, total is N. Caller should size ProgressBar to
    match progress_total().
    """
    do_blur = bool(enable_blur) and float(blur_strength) > 0 and int(blur_window) > 1
    do_grain = bool(enable_grain) and float(grain_strength) > 0
    n = images.shape[0] if images is not None and images.ndim == 4 else 0

    if not do_blur and not do_grain:
        _report(progress_callback, n)
        return images

    chunk = max(1, int(chunk_frames))

    if do_blur and do_grain:
        # Single walk over chunks: blur into out, then grain that chunk.
        # Progress still reported as 2*N units (blur half + grain half) for a
        # clear two-phase bar, or we can do N with one update per finished frame.
        # Prefer one unit per fully finished frame — more intuitive in the UI.
        window = _normalize_window(blur_window)
        weights = build_tmix_weight_list(window, float(blur_strength))
        center = window // 2
        amplitude = float(grain_strength) / 255.0
        generator = _make_generator(images.device, seed)

        out = torch.empty(images.shape, dtype=images.dtype, device=images.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sl = out[start:end]
            _blur_chunk(sl, images, start, end, weights, center)
            _grain_chunk(sl, amplitude, generator)
            _report(progress_callback, end - start)
        return out

    if do_blur:
        return temporal_motion_blur(
            images,
            blur_window,
            blur_strength,
            progress_callback=progress_callback,
            chunk_frames=chunk,
        )

    # Grain only — clone once so we never mutate a shared ComfyUI IMAGE tensor.
    return apply_film_grain(
        images,
        grain_strength,
        seed=seed,
        chunk_frames=chunk,
        inplace=False,
        progress_callback=progress_callback,
    )


def progress_total(
    num_frames: int,
    enable_blur: bool = True,
    blur_window: int = 3,
    blur_strength: float = 0.35,
    enable_grain: bool = True,
    grain_strength: float = 6.0,
) -> int:
    """Total ProgressBar units for a run (one unit per processed frame)."""
    do_blur = bool(enable_blur) and float(blur_strength) > 0 and int(blur_window) > 1
    do_grain = bool(enable_grain) and float(grain_strength) > 0
    if not do_blur and not do_grain:
        return max(1, int(num_frames))  # instant complete still needs total > 0
    return max(1, int(num_frames))
