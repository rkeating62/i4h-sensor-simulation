# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate a linear detector exposure and create an enhanced display image.

Arrays have shape (rows, columns); calibration exposures must match the raw
exposure's acquisition settings. This expects linear counts, not logged data.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class Radiograph:
    # NumPy for CPU processing, CuPy for CUDA processing.
    attenuation: Any  # float32, -log(transmission), no display processing
    display: Any  # uint16, enhanced for viewing
    repaired_pixels: Any  # bool, interpolated detector locations
    clipped_pixels: Any  # bool, transmission at/below the log floor

    def rtk_projection(self):
        """Return an ITK float image (x, y, one projection) for RTK pipelines.

        Pixel spacing, detector origin, and acquisition geometry must be supplied
        separately using the detector's physical calibration.
        """
        import itk

        attenuation = self.attenuation
        if hasattr(attenuation, "__cuda_array_interface__"):
            attenuation = attenuation.get()  # Explicit host transfer for ITK export only.
        return itk.image_from_array(attenuation[np.newaxis, ...])


def _frame(value, shape, name, xp, dtype):
    result = xp.asarray(value, dtype=dtype)
    if result.ndim == 0:
        return result
    if result.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    return result.copy()


def process_radiograph(
    raw: Any,
    *,
    dark=0.0,
    flat=None,
    i0: float | None = None,
    bad_pixel_mask: np.ndarray | None = None,
    saturation_level: float | None = None,
    min_flat_signal: float = 1e-6,
    transmission_floor: float = 1e-6,
    display_mask: np.ndarray | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    median_radius: int = 1,
    equalization: Literal["adaptive", "global", "none"] = "adaptive",
    adaptive_radius: int = 16,
    alpha: float = 0.3,
    beta: float = 0.3,
    invert: bool = False,
    backend: Literal["cpu", "cuda"] = "cpu",
    precision: Literal["float32", "float64"] = "float64",
    percentile_stride: int = 1,
) -> Radiograph:
    """Convert raw counts into calibrated attenuation and a display radiograph.

    T = (raw - dark) / (flat - dark), or (raw - dark) / i0.
    ``flat`` is an open-beam exposure including dark offset. ``i0`` is a
    positive, dark-subtracted open-beam signal; supply exactly one of them.
    Average multiple matched dark/flat frames before calling this function.

    Defects (explicit mask, nonfinite values, invalid flat gain, saturation)
    are replaced with the nearest valid transmission. Low/negative measured
    signals are floored before logging and reported in ``clipped_pixels``.
    Transmission above one is preserved, so attenuation can be negative.

    Median filtering, percentile windowing, and equalization affect display
    only. ``display_mask`` selects the exposed/anatomical region for window
    estimation and global equalization; excluded pixels display as zero.
    The CPU backend uses ITK for median filtering and adaptive equalization
    (install the `rtk` extra). Its alpha/beta parameters blend local equalization,
    sharpening, and pass-through. The CUDA backend uses CuPy and cuCIM, with
    contrast-limited adaptive histogram equalization (CLAHE) on tiles of up to
    `2 * adaptive_radius + 1` pixels per side. Alpha and beta apply only to ITK.
    The CPU backend returns NumPy arrays. The CUDA backend accepts CuPy arrays
    or CUDA tensors supporting DLPack and returns CuPy arrays; detector and
    radiograph images stay on the GPU. Validation may synchronize scalar flags.
    ``precision="float32"`` reduces calibration memory and arithmetic costs for
    interactive previews of simulated signals. Float64 remains the default for
    measured exposures where dark subtraction may require extra precision.
    ``percentile_stride`` samples every Nth row and column for display window
    estimation only; attenuation and output images retain their full shape.
    A sampled ROI with no selected pixels falls back to all selected pixels.
    Higher attenuation is brighter unless ``invert=True``.
    """
    if backend == "cuda":
        try:
            import cupy as xp
            from cupyx.scipy import ndimage
        except ImportError as exc:
            raise RuntimeError("Install the qt-cuda12 or qt-cuda13 extra for CUDA radiographs") from exc
    elif backend == "cpu":
        from scipy import ndimage

        xp = np
    else:
        raise ValueError(f"Unknown radiograph backend: {backend}")

    if backend == "cuda" and not isinstance(raw, np.ndarray) and hasattr(raw, "__dlpack__"):
        raw = xp.from_dlpack(raw)
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if (not isinstance(percentile_stride, int) or isinstance(percentile_stride, bool)
            or percentile_stride < 1):
        raise ValueError("percentile_stride must be a positive integer")
    dtype = xp.float32 if precision == "float32" else xp.float64
    raw = xp.asarray(raw, dtype=dtype)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("raw must be a nonempty 2D array")
    if (flat is None) == (i0 is None):
        raise ValueError("Supply exactly one of flat or i0")
    if not (0 < transmission_floor < 1):
        raise ValueError("transmission_floor must be between 0 and 1")
    if not np.isfinite(min_flat_signal) or min_flat_signal <= 0:
        raise ValueError("min_flat_signal must be positive and finite")
    low, high = percentiles
    if not (0 <= low < high <= 100):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if equalization not in {"adaptive", "global", "none"}:
        raise ValueError("Unknown equalization method")
    if any(not isinstance(r, int) or r < 0 for r in (median_radius, adaptive_radius)):
        raise ValueError("Filter radii must be nonnegative integers")
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError("alpha and beta must lie in [0, 1]")

    dark = _frame(dark, raw.shape, "dark", xp, dtype)
    if flat is not None:
        flat_frame = _frame(flat, raw.shape, "flat", xp, dtype)
        denominator = flat_frame - dark
    else:
        if not np.isfinite(i0) or i0 <= 0:
            raise ValueError("i0 must be positive and finite")
        denominator = i0

    # CuPy follows IEEE floating-point semantics without NumPy's warnings.
    with np.errstate(invalid="ignore", over="ignore"):
        signal = raw - dark
    invalid = (~xp.isfinite(signal) | ~xp.isfinite(denominator)
               | (denominator <= min_flat_signal))
    if saturation_level is not None:
        if not np.isfinite(saturation_level):
            raise ValueError("saturation_level must be finite")
        invalid |= raw >= saturation_level
        if flat is not None:
            invalid |= flat_frame >= saturation_level
    if bad_pixel_mask is not None:
        mask = xp.asarray(bad_pixel_mask, dtype=bool)
        if mask.shape != raw.shape:
            raise ValueError("bad_pixel_mask must match raw.shape")
        invalid |= mask
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        transmission = signal / denominator
    invalid |= ~xp.isfinite(transmission)
    if invalid.any():
        if invalid.all():
            raise ValueError("No valid detector pixels remain")
        nearest = ndimage.distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        transmission[invalid] = transmission[tuple(nearest[:, invalid])]
    clipped = transmission <= transmission_floor
    attenuation = -xp.log(xp.maximum(transmission, transmission_floor))
    attenuation = attenuation.astype(xp.float32)

    roi = None
    if display_mask is not None:
        roi = xp.asarray(display_mask, dtype=bool)
        if roi.shape != raw.shape or not roi.any():
            raise ValueError("display_mask must match raw.shape and select pixels")

    values = attenuation
    if median_radius:
        if backend == "cuda":
            values = ndimage.median_filter(values, size=2 * median_radius + 1, mode="nearest")
        else:
            import itk

            display = itk.median_image_filter(itk.image_from_array(values), radius=median_radius)
            values = itk.array_from_image(display)
    samples = values[::percentile_stride, ::percentile_stride]
    if roi is not None:
        samples = samples[roi[::percentile_stride, ::percentile_stride]]
        if samples.size == 0:
            samples = values[roi]
    lower, upper = xp.percentile(samples, (low, high)).astype(dtype)
    if upper <= lower:
        normalized = xp.zeros(raw.shape, dtype=xp.float32)
    else:
        normalized = xp.clip((values - lower) / (upper - lower), 0, 1)
        if equalization == "adaptive":
            if backend == "cuda":
                if adaptive_radius:
                    try:
                        from cucim.skimage.exposure import equalize_adapthist
                    except ImportError as exc:
                        raise RuntimeError("Install the qt-cuda12 or qt-cuda13 extra for CUDA radiographs") from exc
                    tile_shape = tuple(min(2 * adaptive_radius + 1, size) for size in raw.shape)
                    normalized = equalize_adapthist(normalized, kernel_size=tile_shape)
            else:
                import itk

                enhanced = itk.adaptive_histogram_equalization_image_filter(
                    itk.image_from_array(normalized.astype(np.float32)),
                    radius=adaptive_radius, alpha=alpha, beta=beta,
                )
                normalized = np.clip(itk.array_from_image(enhanced), 0, 1)
        elif equalization == "global":
            # Empirical CDF; equal values receive equal intensities.
            samples = normalized if roi is None else normalized[roi]
            levels, counts = xp.unique(samples, return_counts=True)
            cdf = xp.cumsum(counts, dtype=xp.float64)
            if cdf[-1] > cdf[0]:
                cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
                normalized = xp.interp(normalized, levels, cdf)
    if invert:
        normalized = 1 - normalized
    if roi is not None:
        normalized[~roi] = 0
    return Radiograph(
        attenuation=attenuation,
        display=xp.rint(normalized * 65535).astype(xp.uint16),
        repaired_pixels=invalid,
        clipped_pixels=clipped,
    )
