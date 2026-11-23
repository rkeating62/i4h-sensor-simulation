"""Calibrate a linear detector exposure and create an enhanced display image.

Install: python -m pip install itk-rtk scipy
Arrays have shape (rows, columns); calibration exposures must match the raw
exposure's acquisition settings. This expects linear counts, not logged data.
"""

from dataclasses import dataclass
from typing import Literal

import itk
import numpy as np
from scipy import ndimage


@dataclass
class Radiograph:
    attenuation: np.ndarray  # float32, -log(transmission), no display processing
    display: np.ndarray  # uint16, enhanced for viewing
    repaired_pixels: np.ndarray  # bool, interpolated detector locations
    clipped_pixels: np.ndarray  # bool, transmission at/below the log floor

    def rtk_projection(self):
        """Return an ITK float image (x, y, one projection) for RTK pipelines.

        Pixel spacing, detector origin, and acquisition geometry must be supplied
        separately using the detector's physical calibration.
        """
        return itk.image_from_array(self.attenuation[np.newaxis, ...])


def _frame(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        return np.full(shape, result, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    return result.copy()


def process_radiograph(
    raw: np.ndarray,
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
    ITK adaptive equalization is NOT contrast-limited CLAHE. Its alpha/beta
    parameters blend local equalization, sharpening, and pass-through.
    Higher attenuation is brighter unless ``invert=True``.
    """
    raw = np.asarray(raw, dtype=np.float64)
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

    dark = _frame(dark, raw.shape, "dark")
    if flat is not None:
        flat_frame = _frame(flat, raw.shape, "flat")
        denominator = flat_frame - dark
    else:
        if not np.isfinite(i0) or i0 <= 0:
            raise ValueError("i0 must be positive and finite")
        denominator = np.full(raw.shape, i0)

    with np.errstate(invalid="ignore", over="ignore"):
        signal = raw - dark
    invalid = (~np.isfinite(signal) | ~np.isfinite(denominator)
               | (denominator <= min_flat_signal))
    if saturation_level is not None:
        if not np.isfinite(saturation_level):
            raise ValueError("saturation_level must be finite")
        invalid |= raw >= saturation_level
        if flat is not None:
            invalid |= flat_frame >= saturation_level
    if bad_pixel_mask is not None:
        mask = np.asarray(bad_pixel_mask, dtype=bool)
        if mask.shape != raw.shape:
            raise ValueError("bad_pixel_mask must match raw.shape")
        invalid |= mask
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        transmission = signal / denominator
    invalid |= ~np.isfinite(transmission)
    if invalid.all():
        raise ValueError("No valid detector pixels remain")
    if invalid.any():
        nearest = ndimage.distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        transmission[invalid] = transmission[tuple(nearest[:, invalid])]
    clipped = transmission <= transmission_floor
    attenuation = -np.log(np.maximum(transmission, transmission_floor))
    attenuation = attenuation.astype(np.float32)

    roi = np.ones(raw.shape, dtype=bool)
    if display_mask is not None:
        roi = np.asarray(display_mask, dtype=bool)
        if roi.shape != raw.shape or not roi.any():
            raise ValueError("display_mask must match raw.shape and select pixels")

    display = itk.image_from_array(attenuation)
    if median_radius:
        display = itk.median_image_filter(display, radius=median_radius)
    values = itk.array_from_image(display)
    lower, upper = np.percentile(values[roi], (low, high))
    if upper <= lower:
        normalized = np.zeros(raw.shape, dtype=np.float32)
    else:
        normalized = np.clip((values - lower) / (upper - lower), 0, 1)
        if equalization == "adaptive":
            enhanced = itk.adaptive_histogram_equalization_image_filter(
                itk.image_from_array(normalized.astype(np.float32)),
                radius=adaptive_radius, alpha=alpha, beta=beta,
            )
            normalized = np.clip(itk.array_from_image(enhanced), 0, 1)
        elif equalization == "global":
            # Empirical CDF; equal values receive equal intensities.
            levels, counts = np.unique(normalized[roi], return_counts=True)
            cdf = np.cumsum(counts, dtype=np.float64)
            if cdf[-1] > cdf[0]:
                cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
                normalized = np.interp(normalized, levels, cdf)
    if invert:
        normalized = 1 - normalized
    normalized[~roi] = 0
    return Radiograph(
        attenuation=attenuation,
        display=np.rint(normalized * 65535).astype(np.uint16),
        repaired_pixels=invalid,
        clipped_pixels=clipped,
    )
