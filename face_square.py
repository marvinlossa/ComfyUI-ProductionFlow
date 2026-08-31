"""Detect a face and crop a padded square around it."""

from __future__ import annotations

import os
import sys

import torch

from comfy.model_management import get_torch_device


def _retinaface_class():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI-AutoCropFaces")
    if not os.path.isdir(root):
        raise RuntimeError(
            "ProductionFlow Face Square needs ComfyUI-AutoCropFaces "
            "(RetinaFace) in custom_nodes."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
    return Pytorch_RetinaFace


def _square_box(x1, y1, x2, y2, img_h, img_w, scale, shift):
    face_w = max(1.0, float(x2 - x1))
    face_h = max(1.0, float(y2 - y1))
    cx = float(x1) + face_w * 0.5
    cy = float(y1) + face_h * 0.5
    side = min(int(round(max(face_w, face_h) * float(scale))), img_w, img_h)
    side = max(8, side)
    cy = cy + (0.5 - float(shift)) * side
    x = int(round(cx - side * 0.5))
    y = int(round(cy - side * 0.5))
    x = max(0, min(x, img_w - side))
    y = max(0, min(y, img_h - side))
    return x, y, side


def detect_face_square(image_hwc, scale=1.8, shift=0.42, start_index=0, detector=None):
    """image_hwc: (H, W, C) float [0, 1]. Returns x, y, side."""
    h, w = int(image_hwc.shape[0]), int(image_hwc.shape[1])
    if detector is None:
        detector = _retinaface_class()(device=get_torch_device(), vis_thres=0.6, keep_top_k=20)
    img255 = image_hwc.detach().float() * 255.0
    dets = detector.detect_faces(img255)
    scored = []
    for det in dets:
        if float(det[4]) < detector.vis_thres:
            continue
        x1, y1, x2, y2 = [float(v) for v in det[:4]]
        scored.append((float(det[4]) * max(1.0, (x2 - x1) * (y2 - y1)), x1, y1, x2, y2))
    if not scored:
        side = min(h, w)
        return max(0, (w - side) // 2), max(0, (h - side) // 2), side
    scored.sort(key=lambda t: t[0], reverse=True)
    pick = scored[int(start_index) % len(scored)]
    return _square_box(pick[1], pick[2], pick[3], pick[4], h, w, scale, shift)


def crop_and_region(images, scale=1.8, shift=0.42, start_index=0, detector=None):
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE NHWC, got {tuple(images.shape)}")
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    x, y, side = detect_face_square(images[0], scale, shift, start_index, detector=detector)
    crop = images[:, y:y + side, x:x + side, :].contiguous()
    region = torch.zeros((n, h, w), dtype=images.dtype, device=images.device)
    region[:, y:y + side, x:x + side] = 1.0
    return crop, region, x, y


def paste_mask(mask, canvas_image, x, y):
    if canvas_image.ndim != 4:
        raise ValueError(f"expected IMAGE NHWC, got {tuple(canvas_image.shape)}")
    n, h, w = canvas_image.shape[0], canvas_image.shape[1], canvas_image.shape[2]
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 4:
        m = m.mean(dim=-1)
    if m.shape[0] == 1 and n > 1:
        m = m.expand(n, -1, -1)
    out = torch.zeros((n, h, w), dtype=canvas_image.dtype, device=canvas_image.device)
    x = int(x)
    y = int(y)
    mh, mw = int(m.shape[-2]), int(m.shape[-1])
    dx = max(0, x)
    dy = max(0, y)
    sx = max(0, -x)
    sy = max(0, -y)
    dw = min(w, x + mw) - dx
    dh = min(h, y + mh) - dy
    if dw <= 0 or dh <= 0:
        return out
    src = m.to(device=out.device, dtype=out.dtype)
    out[:, dy:dy + dh, dx:dx + dw] = src[:, sy:sy + dh, sx:sx + dw]
    return out
