"""Bounded worker scheduling and shutdown; requires the viewer CUDA extras."""
import threading
import time

import pytest


@pytest.mark.gpu
def test_worker_keeps_only_latest_pending_state():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('PySide6')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    from examples.xray_interactive_view_qt import RenderWorker

    class Probe(RenderWorker):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.seen = []

        def _render(self, state):
            self.seen.append(state)
            if state == 1:
                self.started.set()
                assert self.release.wait(10)
            return state, state

    worker = Probe()
    worker.start()
    try:
        worker.submit(1, 1)
        assert worker.started.wait(10)
        for generation in range(2, 101):
            worker.submit(generation, generation)
        worker.release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = worker.take_result()
            if result and result[0] == 100:
                assert result[-1] is None
                break
            time.sleep(0.01)
        else:
            pytest.fail('Latest request did not complete')
        assert worker.seen == [1, 100]
    finally:
        worker.release.set()
        worker.stop()
        worker.join(10)
    assert not worker.is_alive()


@pytest.mark.parametrize('width,height,dx,dy', [(3072, 3072, .15, None), (1500, 900, .2, .3)])
def test_preview_preserves_detector_field_of_view(width, height, dx, dy):
    pytest.importorskip('PySide6')
    from examples.xray_interactive_view_qt import preview_geometry
    from xray_simulator import CarmGeometry

    original = CarmGeometry(detector_width_px=width, detector_height_px=height,
                            pixel_spacing_mm=dx, pixel_spacing_y_mm=dy,
                            detector_offset_xy_mm=(12., -8.))
    preview = preview_geometry(original)
    assert (preview.detector_height_px, preview.detector_width_px) == (768, 1024)
    assert preview.detector_width_px * preview.pixel_spacing_mm == pytest.approx(width * dx)
    assert preview.detector_height_px * preview.pixel_spacing_y_mm == pytest.approx(height * (dy or dx))
    assert preview.detector_offset_xy_mm == original.detector_offset_xy_mm
    assert preview.source_to_detector_mm == original.source_to_detector_mm
    assert preview.source_to_isocenter_mm == original.source_to_isocenter_mm


@pytest.mark.gpu
def test_blur_has_same_physical_width_in_preview_and_full_resolution():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('PySide6')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    from examples.xray_interactive_view_qt import RenderWorker
    from xray_simulator import RealismSettings

    worker = RenderWorker()
    cfg = RealismSettings(enabled=True, blur_sigma_px=12, gaussian_sigma=0,
                          poisson_photons=0)
    widths = []
    for shape, scale in [((192, 192), (1., 1.)), ((48, 64), (.25, 1/3))]:
        image = cp.zeros(shape, dtype=cp.float32)
        image[shape[0] // 2, shape[1] // 2] = 1
        blurred = worker._apply_gpu_realism(image, cfg, scale)
        # Second moment of the impulse response, in full-detector pixel units.
        y, x = cp.indices(shape)
        variance = [cp.sum(blurred * ((coord - size // 2) / pitch)**2) / blurred.sum()
                    for coord, size, pitch in zip((y, x), shape, scale)]
        widths.append([float(cp.sqrt(value)) for value in variance])
        assert float(blurred.max()) < 1
        assert float(image.max()) == 1  # Cached sensor remains unmodified.
    assert widths[0] == pytest.approx([12, 12], rel=.002)
    assert widths[1] == pytest.approx(widths[0], rel=.002)
