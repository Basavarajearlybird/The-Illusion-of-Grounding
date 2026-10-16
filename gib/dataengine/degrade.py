"""Axis 2 -- visual ambiguity (the OpenCV degradation bank).

Three parametric families, all applied to the clean image and all written to disk
*once*, content-hashed. Writing degraded images eagerly rather than degrading on
the fly in the eval loop is a deliberate choice: it means a re-run cannot drift,
and it means the human study (Layer 4) can be served the byte-identical images
the models saw, which is what makes the human-model gap legitimate.

The crop family is the surgical one. It crops so that only a target *fraction of
the answer-bearing object's area* remains visible -- not the image centre. That
is why bbox ingestion in Layer 1.1 is mandatory: cropping the centre of a photo
tests nothing in particular, whereas removing 95% of the object the question asks
about tests perception precisely.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from gib.config import IMAGES_DEGRADED_DIR
from gib.schema import Axis, CleanPair, Degradation, Variant, make_variant_id, sha256_bytes
from gib.seeds import rng_for

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Kernel auto-sized from sigma (6-sigma support, forced odd)."""
    if sigma <= 0:
        return img.copy()
    k = int(2 * math.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalised directional line kernel."""
    length = max(1, int(length))
    k = np.zeros((length, length), dtype=np.float32)
    cx = cy = (length - 1) / 2.0
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length * 4):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    total = k.sum()
    if total == 0:
        k[int(cy), :] = 1.0
        total = k.sum()
    return k / total


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    return cv2.filter2D(img, -1, motion_blur_kernel(length, angle_deg))


def crop_to_visible_fraction(
    img: np.ndarray, bbox: list[float], fraction: float, out_size: int = 336
) -> np.ndarray:
    """Crop so ``fraction`` of the answer object's area survives, then resize.

    The crop window is anchored to the object's centre and scaled by
    ``sqrt(fraction)`` in each dimension, so the *area* retained is the requested
    fraction. Some surrounding context is kept (the window is grown to a minimum
    size) because a window smaller than a few pixels would degenerate into a
    colour patch and stop being a perception test at all.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = [float(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        bw, bh = w * 0.5, h * 0.5
        x, y = w * 0.25, h * 0.25

    scale = math.sqrt(max(1e-6, min(1.0, fraction)))
    cw, ch = max(8.0, bw * scale), max(8.0, bh * scale)
    cx, cy = x + bw / 2.0, y + bh / 2.0

    x0 = int(round(max(0, min(w - 1, cx - cw / 2))))
    y0 = int(round(max(0, min(h - 1, cy - ch / 2))))
    x1 = int(round(max(x0 + 1, min(w, cx + cw / 2))))
    y1 = int(round(max(y0 + 1, min(h, cy + ch / 2))))

    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        patch = img
    # INTER_CUBIC on upscale: the model must receive its native resolution, and a
    # nearest-neighbour blow-up would add a blockiness artefact that is a second,
    # uncontrolled degradation.
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# Variant construction
# --------------------------------------------------------------------------- #


def _write_degraded(img: np.ndarray, out_dir: Path, stem: str) -> tuple[str, str]:
    """Encode once, hash the encoded bytes, write if absent. Returns (path, sha)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to encode degraded image {stem}")
    data = buf.tobytes()
    sha = sha256_bytes(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.jpg"
    if not path.exists():
        path.write_bytes(data)
    return str(path), sha


def degrade_image(
    img: np.ndarray,
    kind: Degradation,
    param: float,
    *,
    bbox: list[float] | None = None,
    angle: float = 0.0,
    out_size: int = 336,
) -> np.ndarray:
    if kind is Degradation.GAUSSIAN_BLUR:
        return gaussian_blur(img, param)
    if kind is Degradation.MOTION_BLUR:
        return motion_blur(img, int(param), angle)
    if kind is Degradation.CROP:
        return crop_to_visible_fraction(img, bbox or [], param, out_size)
    return img.copy()


def make_visual_variants(
    pair: CleanPair,
    *,
    gaussian_sigma: list[float],
    motion_kernel: list[int],
    crop_visible: list[float],
    native_resolution: int = 336,
    out_dir: Path | None = None,
    seed_base: int = 20260721,
) -> list[Variant]:
    """13 degradation settings per pair (5 sigma + 4 kernels + 4 crops).

    The retrieved context on this axis is the *correct* ground-truth document.
    That is what isolates the axis: any error here is a perception failure, and
    a model that follows the correct text over a blurry image is exhibiting the
    opposite of Sensory Capture rather than being confused by noise.
    """
    out_dir = Path(out_dir or IMAGES_DEGRADED_DIR)
    img = cv2.imread(pair.image_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    specs: list[tuple[Degradation, float, str, str]] = []
    for s in gaussian_sigma:
        specs.append((Degradation.GAUSSIAN_BLUR, float(s), f"sigma{s:g}", "gauss"))
    for k in motion_kernel:
        specs.append((Degradation.MOTION_BLUR, float(k), f"kernel{int(k)}", "motion"))
    for c in crop_visible:
        specs.append((Degradation.CROP, float(c), f"crop{int(round(c*100))}", "crop"))

    out: list[Variant] = []
    for kind, param, dose, tag in specs:
        # Motion-blur angle is randomised per item but logged, so the direction
        # is not a hidden constant yet the variant stays exactly reproducible.
        angle = 0.0
        if kind is Degradation.MOTION_BLUR:
            angle = round(rng_for("motion_angle", pair.pair_id, param, base=seed_base)
                          .uniform(0.0, 180.0), 2)

        deg = degrade_image(
            img, kind, param, bbox=pair.bbox, angle=angle, out_size=native_resolution
        )
        stem = f"{pair.pair_id}-{tag}-{dose}"
        path, sha = _write_degraded(deg, out_dir, stem)

        out.append(
            Variant(
                pair_id=pair.pair_id,
                variant_id=make_variant_id(pair.pair_id, Axis.VISUAL, dose),
                sub_type=pair.sub_type,
                axis=Axis.VISUAL,
                dose=dose,
                image_path=path,
                image_sha256=sha,
                question=pair.question,
                answer_visual=pair.answer,
                answer_textual="",  # no falsehood is injected on this axis
                retrieved_docs=[pair.ground_truth_doc],
                doc_positions=[0],
                gt_doc_position=0,
                bbox=pair.bbox,
                degradation=kind,
                degradation_param=param,
                motion_angle=angle,
                source=pair.source,
                seed=seed_base,
            )
        )
    return out


def make_combined_variants(
    pair: CleanPair,
    pool,
    *,
    sigma_grid: list[float],
    distractor_grid: list[int],
    hard_ratio: float = 0.5,
    out_dir: Path | None = None,
    seed_base: int = 20260721,
) -> list[Variant]:
    """The visual x noise sub-grid behind the vulnerability heatmap.

    This is the only place the architecture crosses two axes, and it is confined
    to a small subset of pairs on purpose -- the full cross-product would be
    enormous and would buy nothing beyond this one figure.
    """
    from gib.dataengine.hardneg import HardNegativePool  # noqa: F401  (typing only)

    out_dir = Path(out_dir or IMAGES_DEGRADED_DIR)
    img = cv2.imread(pair.image_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    out: list[Variant] = []
    for sigma in sigma_grid:
        if sigma > 0:
            deg = gaussian_blur(img, float(sigma))
            path, sha = _write_degraded(deg, out_dir, f"{pair.pair_id}-gauss-sigma{sigma:g}")
        else:
            path, sha = pair.image_path, pair.image_sha256

        for n_dist in distractor_grid:
            rng = rng_for("combined", pair.pair_id, sigma, n_dist, base=seed_base)
            distractors = pool.take(n_dist, hard_ratio, rng) if n_dist > 0 else []
            docs = [pair.ground_truth_doc] + distractors
            order = list(range(len(docs)))
            rng.shuffle(order)

            out.append(
                Variant(
                    pair_id=pair.pair_id,
                    variant_id=make_variant_id(
                        pair.pair_id, Axis.COMBINED, f"s{sigma:g}", f"d{n_dist}"
                    ),
                    sub_type=pair.sub_type,
                    axis=Axis.COMBINED,
                    dose=f"sigma{sigma:g}|d{n_dist}",
                    image_path=path,
                    image_sha256=sha,
                    question=pair.question,
                    answer_visual=pair.answer,
                    answer_textual=pool.injected_falsehood if n_dist > 0 else "",
                    retrieved_docs=[docs[i] for i in order],
                    doc_positions=order,
                    gt_doc_position=order.index(0),
                    bbox=pair.bbox,
                    degradation=Degradation.GAUSSIAN_BLUR if sigma > 0 else Degradation.NONE,
                    degradation_param=float(sigma),
                    n_distractors=n_dist,
                    source=pair.source,
                    seed=seed_base,
                )
            )
    return out
