"""Calibration invariants and CUDA residency of radiograph processing.

Set XRAY_TEST_CUDA_GL=1 on an NVIDIA desktop to include the Qt interop test.
"""

import os
import numpy as np
import pytest

from xray_simulator.radiograph import process_radiograph


def test_calibration_preserves_physical_attenuation_and_input():
    raw = np.array([[110, 60, 35], [10, 0, 210]], dtype=np.float32)
    original = raw.copy()
    result = process_radiograph(raw, dark=10, flat=110, median_radius=0,
                                equalization="none", percentiles=(0, 100))
    expected = -np.log(np.maximum((raw - 10) / 100, 1e-6))
    np.testing.assert_allclose(result.attenuation, expected, atol=1e-6)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(result.clipped_pixels, raw <= 10)
    assert not result.repaired_pixels.any()
    assert result.display.dtype == np.uint16
    assert result.display[1, 0] == 65535
    assert result.display[1, 2] == 0


def test_invalid_pixels_repaired_and_roi_blacked_out():
    raw = np.array([[1, np.nan, 0.5], [0.2, 0.1, 0.05]])
    roi = np.array([[True, True, True], [False, False, False]])
    result = process_radiograph(raw, i0=1, display_mask=roi,
                                median_radius=0, equalization="global")
    assert result.repaired_pixels[0, 1]
    assert np.isfinite(result.attenuation).all()
    assert (result.display[~roi] == 0).all()


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("equalization", ["none", "global", "adaptive"])
def test_cuda_outputs_stay_on_device(cupy, monkeypatch, equalization):
    cp = cupy
    if equalization == "adaptive":
        pytest.importorskip("cucim.skimage.exposure")
    host = np.linspace(0.05, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
    raw = cp.asarray(host)
    # No full-frame readback is permitted inside the CUDA processing path.
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", lambda *a, **k: pytest.fail("Unexpected host readback"))
        result = process_radiograph(raw, i0=1, backend="cuda", equalization=equalization,
                                    median_radius=0, adaptive_radius=3)
        for value in vars(result).values():
            assert isinstance(value, cp.ndarray)
            assert value.device.id == raw.device.id
    np.testing.assert_allclose(result.attenuation.get(), -np.log(host), atol=1e-6)
    np.testing.assert_array_equal(raw.get(), host)
    if equalization != "adaptive":
        cpu = process_radiograph(host, i0=1, equalization=equalization, median_radius=0)
        np.testing.assert_allclose(result.display.get().astype(float), cpu.display, atol=1)


@pytest.mark.gpu
def test_cuda_accepts_torch_dlpack_on_nondefault_stream(cupy):
    cp = cupy
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch required")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        raw = torch.full((16, 24), 0.25, device="cuda")
        result = process_radiograph(raw, i0=1, backend="cuda", equalization="none",
                                    median_radius=0)
    assert isinstance(result.display, cp.ndarray)
    np.testing.assert_allclose(result.attenuation.get(), -np.log(0.25), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("XRAY_TEST_CUDA_GL") != "1",
                    reason="Set XRAY_TEST_CUDA_GL=1 on an NVIDIA OpenGL desktop")
def test_qt_interop_upload_resize_and_cleanup(cupy):
    from examples.xray_interactive_view_qt import (
        QApplication, CudaImageView, QSurfaceFormat,
    )

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication.instance() or QApplication([])
    views = [CudaImageView(), CudaImageView()]
    try:
        for view in views:
            view.show()
        app.processEvents()
        for shape in [(19, 31), (19, 31), (27, 17)]:
            # Odd widths also exercise the GL pixel unpack alignment.
            host = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) * 71
            with cupy.cuda.Stream(non_blocking=True):
                for view in views:
                    view.set_image(cupy.asarray(host))
            app.processEvents()
            for view in views:
                view.makeCurrent()
                try:
                    gl = view.gl
                    gl.glBindTexture(gl.GL_TEXTURE_2D, view._texture)
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
                    # Readback is intentional here, to verify the device upload.
                    actual = np.empty(shape, dtype=np.uint16)
                    gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RED, gl.GL_UNSIGNED_SHORT, actual)
                    np.testing.assert_array_equal(actual, host)
                    assert gl.glGetError() == gl.GL_NO_ERROR
                finally:
                    view.doneCurrent()
        for view in views:
            view.cleanup()
            assert view._resource is None
            assert view._texture == view._pbo == view._program == view._vao == 0
            view.cleanup()  # Qt may also deliver aboutToBeDestroyed after close.
    finally:
        for view in views:
            if view.isValid():
                view.cleanup()
            view.close()
        app.processEvents()


def test_preview_precision_and_subsampled_window():
    raw = np.linspace(.01, 1, 48 * 64, dtype=np.float32).reshape(48, 64)
    options = dict(i0=1, median_radius=0, equalization='none', percentiles=(1, 99))
    reference = process_radiograph(raw, **options)
    preview = process_radiograph(raw, **options, precision='float32', percentile_stride=4)
    np.testing.assert_allclose(preview.attenuation, reference.attenuation, atol=1e-6)
    lower, upper = np.percentile(preview.attenuation[::4, ::4], (1, 99)).astype(np.float32)
    expected = np.rint(np.clip((preview.attenuation - lower) / (upper - lower), 0, 1) * 65535)
    np.testing.assert_array_equal(preview.display, expected.astype(np.uint16))
    assert preview.display.shape == raw.shape


def test_subsampling_falls_back_for_sparse_roi():
    raw = np.linspace(.1, 1, 64, dtype=np.float32).reshape(8, 8)
    roi = np.zeros_like(raw, dtype=bool)
    roi[1, 1:4] = True
    options = dict(i0=1, median_radius=0, equalization='none', display_mask=roi)
    full = process_radiograph(raw, **options)
    sampled = process_radiograph(raw, **options, percentile_stride=4)
    np.testing.assert_array_equal(sampled.display, full.display)


@pytest.mark.parametrize('kwargs', [dict(precision='float16'), dict(percentile_stride=0),
                                    dict(percentile_stride=1.5), dict(percentile_stride=True)])
def test_invalid_preview_options(kwargs):
    with pytest.raises(ValueError):
        process_radiograph(np.ones((4, 4)), i0=1, **kwargs)
