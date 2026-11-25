#!/usr/bin/env python3
"""Qt-based interactive X-ray simulation viewer.

This GUI renders fluoroscopy frames from a preprocessed CT volume and exposes
controls for geometry, pose, and simple display/noise effects.
Install the qt-cuda12 or qt-cuda13 project extra for CUDA processing and display.
Requires an NVIDIA-backed desktop OpenGL 3.3 context on the simulation GPU.
Sensor and radiograph pixels stay on the GPU; only pose parameters are uploaded.

Supported load targets:
- a cached preprocessed volume directory containing ``mu_volume.npy`` and ``metadata.json``
- a ``.nii`` or ``.nii.gz`` volume
- a 3D ``.npy`` volume
- a 2D image file, which is converted into a simple pseudo-volume for demos
"""

from __future__ import annotations

import os
import ctypes
import threading
from dataclasses import dataclass, replace
import math
import sys
from pathlib import Path

import numpy as np

from xray_simulator import (
    CarmGeometry,
    HuToMuMapping,
    PreprocessingSettings,
    PreprocessedVolume,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
    VolumePreprocessor,
    xray_simulator,
)
from xray_simulator.radiograph import process_radiograph
from xray_simulator.geometry import euler_zxy_to_matrix, matrix_to_euler_zxy

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QDoubleValidator, QIntValidator, QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PySide6"
except ImportError:
    try:
        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtGui import QDoubleValidator, QIntValidator, QSurfaceFormat
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
        QT_API = "PyQt6"
    except ImportError:
        try:
            from PyQt5.QtCore import QTimer, Qt
            from PyQt5.QtGui import QDoubleValidator, QIntValidator, QSurfaceFormat
            from PyQt5.QtWidgets import (
                QOpenGLWidget,
                QApplication,
                QCheckBox,
                QFileDialog,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QSlider,
                QVBoxLayout,
                QWidget,
            )
            QT_API = "PyQt5"
        except ImportError as exc:
            raise SystemExit(
                "Qt bindings not found. Install PySide6, PyQt6, or PyQt5 to run "
                "xray_interactive_view_qt.py."
            ) from exc

_PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("FLUOROSIM_OUTPUT_DIR", str(_PROJECT_DIR / "output"))).expanduser()
CACHE_DIR = Path(os.environ.get("FLUOROSIM_CACHE_DIR", str(OUTPUT_DIR / "cache"))).expanduser()

DEFAULT_GEOMETRY = CarmGeometry(
    detector_width_px=3072,
    detector_height_px=3072,
    pixel_spacing_mm=0.15,
    source_to_detector_mm=2220.0,
    source_to_isocenter_mm=2180.0,
)

DEFAULT_PHYSICS = XrayPhysics(
        step_mm=0.5,                    # Ray-march step size (smaller = more accurate)
        i0=1.0,                         # Unattenuated intensity
        normalize=False,                # Preserve linear detector intensities for calibration
        invert=False,                   # Radiograph processing makes bone bright for display
)

DEFAULT_HU_TO_MU = HuToMuMapping(
    hu_min = -1000.0,                   # Defaults to -1000
    hu_max = 3000.0,                    # Defaults to 3000 (dense bone)
    mu_min = 0.0,                       # Defaults to 0.0
    mu_max = 0.02,                      # Defaults to 0.02 (approximate mu for bone at typical X-ray energies
)

DEFAULT_PREPROCESSING_SETTINGS = PreprocessingSettings(
    hu_clip_min = -1024.0,              # Defaults to -1024 (air)
    hu_clip_max = 3071.0,               # Defaults to 3071 (bone)
    clip_hu = True,
    hu_to_mu = DEFAULT_HU_TO_MU,
)

try:
    ORIENTATION_HORIZONTAL = Qt.Orientation.Horizontal
    ALIGN_RIGHT_VCENTER = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
except AttributeError:
    ORIENTATION_HORIZONTAL = Qt.Horizontal
    ALIGN_RIGHT_VCENTER = Qt.AlignRight | Qt.AlignVCenter


def create_synthetic_volume() -> PreprocessedVolume:
    # Create synthetic volume for demo
    print("\n  Creating synthetic sphere volume for demo...")
    import numpy as np
    shape = (128, 256, 256)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

    # Sphere with bone density in air
    hu_volume = np.where(dist < 50, 800.0, -900.0).astype(np.float32)

    preprocessor = VolumePreprocessor.from_numpy(
        hu_volume,
        spacing_zyx_mm=(1.0, 0.5, 0.5)
    )
    return preprocessor.preprocess(output_dir=CACHE_DIR)


def load_cached_volume() -> PreprocessedVolume:
    if (CACHE_DIR / "mu_volume.npy").exists():
        print(f"  Loading cached volume from: {CACHE_DIR}")
        return PreprocessedVolume.load(CACHE_DIR)
    else:
        return create_synthetic_volume()


def load_ct_volume_from_path(path: Path) -> PreprocessedVolume:
     # Check if we have a cached preprocessed volume
    if path.exists():
        print(f"  Loading NIfTI from: {path}")
        preprocessor = VolumePreprocessor.from_nifti(nifti_path=path, settings=DEFAULT_PREPROCESSING_SETTINGS)
        print(f"  Shape: {preprocessor.shape}")
        print(f"  Spacing: {preprocessor.spacing_zyx_mm} mm")
        print(f"  HU range: {preprocessor.hu_range}")
        return preprocessor.preprocess(output_dir=CACHE_DIR)
    else:
        print("\n  ERROR: CT data not found at:")
        print(f"    NIfTI: {path}")
        print("     Will use cached or synthetic volume.")
        return load_cached_volume()


def _cuda_result(result):
    """Unpack cuda-python's status-first return values."""
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA–OpenGL interop failed: {status}")
    return values


class CudaImageView(QOpenGLWidget):
    """Display a CuPy uint16 image through a CUDA-registered pixel buffer.

    Two registered pixel buffers rotate on a dedicated CUDA upload stream.
    CUDA events retain sources until unmap completes; GL fences protect buffers
    until texture transfers finish. Completion is polled without CPU waits.
    Resources belong to this widget's GL context. Only teardown waits for work.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self._slots = []
        self._pending_image = None
        self._retired_sources = []
        self._upload_stream = None
        self._texture = self._program = self._vao = 0
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(5)
        self._poll_timer.timeout.connect(self.update)
        self._shape = None
        self._image = None
        self._error = None

    def initializeGL(self):  # noqa: N802
        try:
            import cupy as cp
            from cuda.bindings import runtime
            from OpenGL import GL
            from OpenGL.GL.shaders import compileProgram, compileShader

            self.cp, self.cuda, self.gl = cp, runtime, GL
            self._upload_stream = cp.cuda.Stream(non_blocking=True)
            self._slots = [dict(pbo=0, resource=None, shape=None, event=None,
                                source=None, fence=None, aspect=None) for _ in range(2)]
            self.context().aboutToBeDestroyed.connect(self.cleanup)
            self._program = compileProgram(
                compileShader("""#version 330 core
                    out vec2 uv;
                    void main() {
                        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                        uv = vec2(p.x, 1.0 - p.y);
                        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                    }""", GL.GL_VERTEX_SHADER),
                compileShader("""#version 330 core
                    in vec2 uv;
                    uniform sampler2D image;
                    out vec4 color;
                    void main() {
                        float gray = texture(image, uv).r;
                        color = vec4(gray, gray, gray, 1.0);
                    }""", GL.GL_FRAGMENT_SHADER),
            )
            self._vao = GL.glGenVertexArrays(1)
            self._error = None
        except Exception as exc:
            self._error = exc

    def _release_image(self):
        # Blocking teardown is intentional; normal presentation only queries
        # events/fences. Never free a source or registered buffer still in use.
        for source in self._retired_sources + ([self._pending_image] if self._pending_image else []):
            source[1].synchronize()
        self._retired_sources = []
        for slot in self._slots:
            if slot["event"] is not None:
                slot["event"].synchronize()
        if self._slots:
            self.gl.glFinish()
        for slot in self._slots:
            if slot["fence"] is not None:
                self.gl.glDeleteSync(slot["fence"])
            if slot["resource"] is not None:
                _cuda_result(self.cuda.cudaGraphicsUnregisterResource(slot["resource"]))
            if slot["pbo"]:
                self.gl.glDeleteBuffers(1, [slot["pbo"]])
        self._slots = []
        self._pending_image = None
        if self._texture:
            self.gl.glDeleteTextures([self._texture])
            self._texture = 0
        self._shape = None

    def set_image(self, image, *, display_aspect=None):
        """Queue the latest device image; paintGL submits and polls uploads."""
        if not self.isValid():
            raise RuntimeError("A working NVIDIA OpenGL context is required")
        if self._error is not None:
            raise RuntimeError(f"Cannot present CUDA image: {self._error}")
        cp = self.cp
        if not isinstance(image, cp.ndarray) or image.ndim != 2 or image.dtype != cp.uint16:
            raise ValueError("Display expects a 2D CuPy uint16 image")
        # Record readiness on the caller's stream; no CPU wait. Worker images
        # are already complete, but this also supports other CUDA producers.
        with image.device:
            image = cp.ascontiguousarray(image)
            ready = cp.cuda.Event(disable_timing=True)
            ready.record()
        if self._pending_image is not None:
            self._retired_sources.append(self._pending_image)
        self._pending_image = (image, ready, display_aspect or image.shape[1] / image.shape[0])
        self._poll_timer.start()
        self.update()

    def _pump_uploads(self):
        cp, GL = self.cp, self.gl
        self._retired_sources = [item for item in self._retired_sources if not item[1].done]
        # Complete uploads in submission order to avoid presenting older frames
        # after newer ones. Slots are rotated after each submission.
        for slot in self._slots:
            if slot["event"] is not None:
                if not slot["event"].done:
                    continue
                image = slot["source"]
                height, width = image.shape
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                if not self._texture:
                    self._texture = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
                if self._shape != image.shape:
                    for parameter in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
                        GL.glTexParameteri(GL.GL_TEXTURE_2D, parameter, GL.GL_LINEAR)
                    for parameter in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
                        GL.glTexParameteri(GL.GL_TEXTURE_2D, parameter, GL.GL_CLAMP_TO_EDGE)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16, width, height,
                                    0, GL.GL_RED, GL.GL_UNSIGNED_SHORT, None)
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, slot["pbo"])
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                                  GL.GL_RED, GL.GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                slot["fence"] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
                GL.glFlush()
                self._shape = image.shape
                self._image = image
                self._display_aspect = slot["aspect"]
                slot["source"] = slot["event"] = None
            if slot["fence"] is not None:
                status = GL.glClientWaitSync(slot["fence"], 0, 0)
                if status == GL.GL_WAIT_FAILED:
                    raise RuntimeError("OpenGL upload fence failed")
                if status in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
                    GL.glDeleteSync(slot["fence"])
                    slot["fence"] = None
        free = next((slot for slot in self._slots
                     if slot["event"] is None and slot["fence"] is None), None)
        if self._pending_image is not None and free is not None:
            image, ready, aspect = self._pending_image
            # Keep the latest queued image alive until its producer completes.
            if ready.done:
                slot = free
                with image.device, self._upload_stream:
                    if slot["shape"] != image.shape:
                        if slot["resource"] is not None:
                            _cuda_result(self.cuda.cudaGraphicsUnregisterResource(slot["resource"]))
                            slot["resource"] = None
                        if not slot["pbo"]:
                            slot["pbo"] = GL.glGenBuffers(1)
                        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, slot["pbo"])
                        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, image.nbytes, None, GL.GL_STREAM_DRAW)
                        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                        slot["resource"], = _cuda_result(self.cuda.cudaGraphicsGLRegisterBuffer(
                            int(slot["pbo"]), self.cuda.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
                        slot["shape"] = image.shape
                    stream = self._upload_stream
                    slot["source"], slot["aspect"] = image, aspect
                    _cuda_result(self.cuda.cudaGraphicsMapResources(1, slot["resource"], stream.ptr))
                    try:
                        ptr, size = _cuda_result(self.cuda.cudaGraphicsResourceGetMappedPointer(slot["resource"]))
                        if size < image.nbytes:
                            raise RuntimeError("Mapped buffer is smaller than the image")
                        cp.cuda.runtime.memcpyAsync(int(ptr), image.data.ptr, image.nbytes,
                                                   cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                    finally:
                        _cuda_result(self.cuda.cudaGraphicsUnmapResources(1, slot["resource"], stream.ptr))
                        slot["event"] = cp.cuda.Event(disable_timing=True)
                        slot["event"].record(stream)
                self._pending_image = None
                self._slots = [item for item in self._slots if item is not slot]
                self._slots.append(slot)
        if not self._retired_sources and self._pending_image is None and all(slot["event"] is None and slot["fence"] is None
                                              for slot in self._slots):
            self._poll_timer.stop()

    def paintGL(self):  # noqa: N802
        if not hasattr(self, "gl"):
            return
        GL = self.gl
        GL.glClearColor(0.06, 0.06, 0.06, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._error is not None:
            return
        try:
            self._pump_uploads()
        except Exception as exc:
            self._error = exc
            self._poll_timer.stop()
            if hasattr(self.window(), "statusBar"):
                self.window().statusBar().showMessage(f"Display failed: {exc}")
            return
        if self._shape is None:
            return
        rows, cols = self._shape
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        aspect = self._display_aspect
        vw = min(width, round(height * aspect))
        vh = min(height, round(width / aspect))
        GL.glViewport((width - vw) // 2, (height - vh) // 2, vw, vh)
        GL.glUseProgram(self._program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "image"), 0)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def cleanup(self):
        self._poll_timer.stop()
        self.makeCurrent()
        try:
            self._release_image()
            if self._vao:
                self.gl.glDeleteVertexArrays(1, [self._vao])
                self._vao = 0
            if self._program:
                self.gl.glDeleteProgram(self._program)
                self._program = 0
            self._image = None
        finally:
            self.doneCurrent()


class SliderRow(QWidget):
    """A labeled slider with value display."""

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        scale: float = 1.0,
        parent: QWidget | None = None,
        decimals: int = 2,
    ) -> None:
        super().__init__(parent)
        self.scale = scale
        self.decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.slider = QSlider(ORIENTATION_HORIZONTAL)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(80)
        self.value_label.setAlignment(ALIGN_RIGHT_VCENTER)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._sync_label)
        self._sync_label(value)

    def _sync_label(self, value: int) -> None:
        self.value_label.setText(f"{value / self.scale:.{self.decimals}f}")

    def value(self) -> float:
        return self.slider.value() / self.scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round(value * self.scale)))


class IntegerInputRow(QWidget):
    """A labeled integer text input that keeps the last committed value."""

    def __init__(
        self,
        label: str,
        value: int,
        minimum: int,
        maximum: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = int(value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.input = QLineEdit(str(self._value))
        self.input.setValidator(QIntValidator(minimum, maximum, self))
        self.input.setAlignment(ALIGN_RIGHT_VCENTER)
        self.input.setMinimumWidth(80)
        self.input.setMaximumWidth(120)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.input)

        self.input.editingFinished.connect(self._commit_text)

    def _commit_text(self) -> None:
        text = self.input.text().strip()

        if text and self.input.hasAcceptableInput():
            self._value = int(text)
        else:
            self.input.setText(str(self._value))

    def value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        self._value = int(value)
        self.input.setText(str(self._value))


class FloatInputRow(QWidget):
    """A labeled floating-point text input that keeps the last committed value."""

    def __init__(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int = 6,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = float(value)
        self._decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.input = QLineEdit(self._format_value(self._value))
        self.input.setValidator(QDoubleValidator(minimum, maximum, decimals, self))
        self.input.setAlignment(ALIGN_RIGHT_VCENTER)
        self.input.setMinimumWidth(80)
        self.input.setMaximumWidth(120)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.input)

        self.input.editingFinished.connect(self._commit_text)

    def _format_value(self, value: float) -> str:
        return f"{value:.{self._decimals}f}".rstrip("0").rstrip(".")

    def _commit_text(self) -> None:
        text = self.input.text().strip()

        if text and self.input.hasAcceptableInput():
            self._value = float(text)
        self.input.setText(self._format_value(self._value))

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = float(value)
        self.input.setText(self._format_value(self._value))


def viewer_rotation(angles_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compose world-axis slider offsets with the initial AP orientation.

    Offsets use Rz @ Rx @ Ry; the AP basis is applied separately on the right.
    Adding -90 directly to rx would couple the Y and Z sliders at zero pose.
    Convert the final matrix to the renderer's ZXY representation only once.
    """
    offsets = euler_zxy_to_matrix(tuple(math.radians(angle) for angle in angles_deg))
    initial = euler_zxy_to_matrix((-math.pi / 2, 0.0, 0.0))
    return matrix_to_euler_zxy(offsets @ initial)


def preview_geometry(geometry: CarmGeometry) -> CarmGeometry:
    """Use 768 rows x 1024 columns over the same physical detector area."""
    dy = geometry.pixel_spacing_y_mm
    if dy is None:
        dy = geometry.pixel_spacing_mm
    return replace(
        geometry, detector_width_px=1024, detector_height_px=768,
        pixel_spacing_mm=geometry.pixel_spacing_mm * geometry.detector_width_px / 1024,
        pixel_spacing_y_mm=dy * geometry.detector_height_px / 768,
    )


@dataclass(frozen=True)
class RenderState:
    volume: PreprocessedVolume
    geometry: CarmGeometry
    realism: RealismSettings
    rotation: tuple
    translation: tuple
    preview: bool = False
    blur_scale_yx: tuple[float, float] = (1.0, 1.0)

    def matches(self, other):
        return (self.volume is other.volume and self.geometry == other.geometry
                and self.realism == other.realism and self.rotation == other.rotation
                and self.translation == other.translation
                and self.preview == other.preview
                and self.blur_scale_yx == other.blur_scale_yx)


class RenderWorker(threading.Thread):
    """Own the renderer and CUDA processing; exchange bounded mailboxes with Qt.

    Only one pending request and one completed result are retained. CUDA work
    finishes on this thread before publishing arrays, so the GUI can use its
    own stream safely. No QWidget or OpenGL operation runs here.
    """

    def __init__(self):
        super().__init__(name="XrayRenderWorker", daemon=False)
        self._condition = threading.Condition()
        self._pending = self._result = None
        self._stopping = False
        self.simulator = None
        self._sensor_image = None
        self._sensor_state = None

    def submit(self, generation, state):
        with self._condition:
            self._pending = (generation, state)
            self._condition.notify()

    def take_result(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def stop(self):
        with self._condition:
            self._stopping = True
            self._pending = self._result = None
            self._condition.notify()

    def run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    generation, state = self._pending
                    self._pending = None
                try:
                    import cupy as cp

                    # A separate stream prevents GUI uploads from waiting for
                    # the worker's next frame on the default stream.
                    with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
                        raw, display = self._render(state)
                    result = (generation, state, raw, display, None)
                except Exception as exc:
                    result = (generation, state, None, None, str(exc))
                with self._condition:
                    if not self._stopping:
                        self._result = result
        finally:
            # Destroy Slang resources on their owning thread after GPU work.
            self._sensor_image = None
            self.simulator = None

    def _render(self, state):
        import cupy as cp

        if self.simulator is None or self.simulator.volume is not state.volume:
            config = SimulatorConfig(geometry=state.geometry, physics=DEFAULT_PHYSICS,
                                     realism=RealismSettings(enabled=False))
            self.simulator = xray_simulator(state.volume, config)
            self._sensor_image = None
        elif self.simulator.config.geometry != state.geometry:
            self.simulator.update_geometry(state.geometry)
            self._sensor_image = None
        previous = self._sensor_state
        if (self._sensor_image is None or previous is None
                or previous.rotation != state.rotation or previous.translation != state.translation):
            batch = self.simulator.renderer.render_batch(
                np.asarray([state.rotation], dtype=np.float32),
                np.asarray([state.translation], dtype=np.float32), as_torch=True)
            if not batch.is_cuda:
                raise RuntimeError("The viewer requires the CUDA rendering backend")
            self._sensor_image = cp.from_dlpack(batch[0])
            self._sensor_state = state
        sensor = self._apply_gpu_realism(self._sensor_image, state.realism, state.blur_scale_yx)
        radiograph = process_radiograph(
            sensor, i0=self.simulator.config.physics.i0, backend="cuda",
            equalization="none" if state.preview else "adaptive",
            precision="float32" if state.preview else "float64",
            percentile_stride=4 if state.preview else 1,
            median_radius=0, adaptive_radius=16, percentiles=(0.1, 99.9))
        raw = self._format_raw_for_display(sensor)
        cp.cuda.get_current_stream().synchronize()
        return raw, radiograph.display

    def _apply_gpu_realism(self, image, cfg, blur_scale_yx=(1.0, 1.0)):
        """Apply the viewer's detector noise and blur controls on the GPU."""
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter

        if not cfg.enabled:
            return image
        out = cp.maximum(image * cfg.gain + cfg.bias, 0)
        rng = cp.random.RandomState(cfg.seed)
        if cfg.poisson_photons > 0:
            out = rng.poisson(out * cfg.poisson_photons).astype(cp.float32) / cfg.poisson_photons
        if cfg.gaussian_sigma > 0:
            out = cp.maximum(out + rng.normal(0, cfg.gaussian_sigma, out.shape).astype(cp.float32), 0)
        if cfg.blur_sigma_px > 0:
            # Slider sigma is measured in full-resolution detector pixels.
            # Preview pixels cover a larger area, independently on each axis.
            sigma = tuple(cfg.blur_sigma_px * scale for scale in blur_scale_yx)
            out = gaussian_filter(out, sigma)
        return out

    def _format_raw_for_display(self, image):
        """Stretch linear intensities to uint16 using device operations only."""
        import cupy as cp

        low, high = image.min(), image.max()
        span = high - low
        normalized = cp.where(span > 0, (image - low) / cp.maximum(span, 1e-20), 0)
        return cp.rint(cp.clip(1 - normalized, 0, 1) * 65535).astype(cp.uint16)


class XraySimulatorWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        # Let Qt compose valid initial GL frames before the compositor reveals
        # the native window. This avoids displaying stale desktop pixels on X11.
        self.setWindowOpacity(0.0)
        self._startup_painted_views = set()
        self.setAutoFillBackground(True)
        self.setWindowTitle(f"X-Ray Simulation Viewer ({QT_API})")
        self.resize(1800, 950)

        self.volume: PreprocessedVolume = load_cached_volume()
        self.realism_enabled = False
        self._last_state = None
        self._generation = 0
        self._worker = RenderWorker()
        self._worker.start()

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render_current_view)

        self._initializing = True
        self._presented_views = set()
        self._initial_frame_status = None
        self._build_ui()
        for control in self._startup_controls:
            control.setEnabled(False)
        for view in (self.raw_image_view, self.image_view):
            view.frameSwapped.connect(lambda view=view: self._reveal_after_first_paint(view))
            view.frameSwapped.connect(lambda view=view: self._initial_frame_presented(view))
        self._drag_sliders = [row.slider for group in (
            self.geometry_sliders, self.pose_sliders, self.realism_sliders
        ) for row in group.values()]
        for slider in self._drag_sliders:
            slider.sliderPressed.connect(self.render_current_view)
            slider.sliderReleased.connect(lambda slider=slider: self._slider_released(slider))
        self._set_status("Simulation is initializing…")
        self.render_timer.start(33)  # Start rendering at 30 fps immediately after UI is shown

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        image_views = QWidget()
        image_layout = QHBoxLayout(image_views)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.raw_image_view = CudaImageView()
        self.image_view = CudaImageView()
        for title, view in (
            ("Simulator output (linear intensity)", self.raw_image_view),
            ("Processed radiograph", self.image_view),
        ):
            panel = QGroupBox(title)
            panel_layout = QVBoxLayout(panel)
            panel_layout.addWidget(view)
            image_layout.addWidget(panel, 1)
        root.addWidget(image_views, 4)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)
        root.addWidget(sidebar, 2)

        load_group = QGroupBox("Data")
        load_layout = QVBoxLayout(load_group)
        self.load_button = QPushButton("Load CT Volume")
        self.load_button.clicked.connect(self.load_ct_volume)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(load_group)

        geometry_group = QGroupBox("Geometry")
        geometry_layout = QVBoxLayout(geometry_group)
        self.geometry_sliders: dict[str, SliderRow] = {
            "sdd": SliderRow("Source-Detector [mm]", 300, 3000, int(DEFAULT_GEOMETRY.source_to_detector_mm), decimals=0),
            "sid": SliderRow("Source-Iso [mm]", 250, 2950, int(DEFAULT_GEOMETRY.source_to_isocenter_mm), decimals=0),
        }
        for slider in self.geometry_sliders.values():
            geometry_layout.addWidget(slider)
        self.detector_inputs: dict[str, IntegerInputRow] = {
            "width": IntegerInputRow("Detector Width [px]", DEFAULT_GEOMETRY.detector_width_px, 1, 8192),
            "height": IntegerInputRow("Detector Height [px]", DEFAULT_GEOMETRY.detector_height_px, 1, 8192),
        }
        for detector_input in self.detector_inputs.values():
            geometry_layout.addWidget(detector_input)
        self.pixel_spacing_input = FloatInputRow(
            "Pixel Spacing [mm]",
            DEFAULT_GEOMETRY.pixel_spacing_mm,
            0.001,
            100.0,
        )
        geometry_layout.addWidget(self.pixel_spacing_input)
        sidebar_layout.addWidget(geometry_group)

        pose_group = QGroupBox("Pose")
        pose_layout = QVBoxLayout(pose_group)
        self.pose_sliders: dict[str, SliderRow] = {
            "rx": SliderRow("X Rotation [deg]", -180, 180, 0, decimals=0),
            "ry": SliderRow("Y Rotation [deg]", -180, 180, 0, decimals=0),
            "rz": SliderRow("Z Rotation [deg]", -180, 180, 0, decimals=0),
            "tx": SliderRow("X Translation [mm]", -1500, 1500, 0, decimals=0),
            "ty": SliderRow("Y Translation [mm]", -1500, 1500, 0, decimals=0),
            "tz": SliderRow("Z Translation [mm]", -1500, 1500, 0, decimals=0),
        }
        for slider in self.pose_sliders.values():
            pose_layout.addWidget(slider)
        sidebar_layout.addWidget(pose_group)

        realism_group = QGroupBox("Realism")
        realism_layout = QVBoxLayout(realism_group)
        self.realism_checkbox = QCheckBox("Enable Realism")
        self.realism_checkbox.toggled.connect(self.toggle_realism)
        realism_layout.addWidget(self.realism_checkbox)

        self.realism_sliders_group = QGroupBox("Realism Sliders")
        realism_sliders_layout = QVBoxLayout(self.realism_sliders_group)
        self.realism_sliders: dict[str, SliderRow] = {
            "poisson": SliderRow("Poisson Noise [photons]", 0, 100, 0, scale=0.01, decimals=0),
            "gaussian": SliderRow("Gaussian Noise Sigma", 0, 100, 0, scale=400.0),
            "blur": SliderRow("Gaussian Blur [px]", 0, 20, 0, decimals=0),
        }
        for slider in self.realism_sliders.values():
            realism_sliders_layout.addWidget(slider)
        self.realism_sliders_group.setEnabled(False)
        realism_layout.addWidget(self.realism_sliders_group)
        sidebar_layout.addWidget(realism_group)

        actions = QHBoxLayout()
        self.reset_button = QPushButton("Reset Controls")
        self.reset_button.clicked.connect(self.reset_controls)
        actions.addWidget(self.reset_button)
        sidebar_layout.addLayout(actions)
        sidebar_layout.addStretch(1)
        self._startup_controls = (
            self.load_button, geometry_group, pose_group, realism_group, self.reset_button,
        )

    def _reveal_after_first_paint(self, view) -> None:
        """Reveal the GUI after both GL backgrounds have reached composition."""
        if self.windowOpacity() >= 1.0:
            return
        self._startup_painted_views.add(view)
        if len(self._startup_painted_views) == 2:
            QTimer.singleShot(0, lambda: self.setWindowOpacity(1.0))

    def _initial_frame_presented(self, view) -> None:
        """Unlock controls after both image widgets have presented real pixels."""
        if not self._initializing or self._initial_frame_status is None or view._image is None:
            return
        self._presented_views.add(view)
        if len(self._presented_views) != 2:
            return
        self._initializing = False
        for control in self._startup_controls:
            control.setEnabled(True)
        self._set_status(self._initial_frame_status)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def geometry_controls(self) -> CarmGeometry:
        return CarmGeometry(
            detector_width_px=self.detector_inputs["width"].value(),
            detector_height_px=self.detector_inputs["height"].value(),
            pixel_spacing_mm=self.pixel_spacing_input.value(),
            source_to_detector_mm=self.geometry_sliders["sdd"].value(),
            source_to_isocenter_mm=self.geometry_sliders["sid"].value(),
        )
    
    def realism_controls(self) -> RealismSettings:
        if not self.realism_enabled:
            return RealismSettings(
                enabled=False,
                gaussian_sigma=0.0,
                poisson_photons=0.0,
                blur_sigma_px=0.0,
                seed=0,
            )

        poisson_level = self.realism_sliders["poisson"].value()
        gaussian_level = self.realism_sliders["gaussian"].value()
        blur_level = self.realism_sliders["blur"].value()

        return RealismSettings(
            enabled=True,
            gaussian_sigma=gaussian_level,
            poisson_photons=poisson_level,
            blur_sigma_px=blur_level,
            seed=0,
        )

    def rotation_controls(self) -> tuple[float, float, float]:
        return viewer_rotation(tuple(self.pose_sliders[key].value() for key in ("rx", "ry", "rz")))

    def translation_controls(self) -> tuple[float, float, float]:
        return (
            self.pose_sliders["tx"].value(),
            self.pose_sliders["ty"].value(),
            self.pose_sliders["tz"].value(),
        )

    def _slider_released(self, slider) -> None:
        """Snap rotation landmarks before requesting the final-resolution frame."""
        if any(slider is self.pose_sliders[key].slider for key in ("rx", "ry", "rz")):
            for target in (-90, 0, 90):
                if abs(slider.value() - target) <= 2:
                    slider.setValue(target)
                    break
        self.render_current_view()

    def render_current_view(self) -> None:
        """Poll completed GPU images and replace the worker's pending request."""
        sid = self.geometry_sliders["sid"].value()
        if self.geometry_sliders["sdd"].value() <= sid:
            self.geometry_sliders["sdd"].set_value(sid + 1)
        preview = any(slider.isSliderDown() for slider in self._drag_sliders)
        geometry = self.geometry_controls()
        blur_scale_yx = (1.0, 1.0)
        if preview:
            full_geometry = geometry
            geometry = preview_geometry(geometry)
            blur_scale_yx = (
                geometry.detector_height_px / full_geometry.detector_height_px,
                geometry.detector_width_px / full_geometry.detector_width_px,
            )
        state = RenderState(self.volume, geometry, self.realism_controls(),
                            self.rotation_controls(), self.translation_controls(), preview, blur_scale_yx)
        if self._last_state is None or not state.matches(self._last_state):
            self._generation += 1
            self._last_state = state
            self._worker.submit(self._generation, state)
        result = self._worker.take_result()
        if result is None:
            return
        generation, state, raw, display, error = result
        if error is not None:
            self.render_timer.stop()
            self._set_status(f"Render failed: {error}")
            QMessageBox.critical(self, "Render Error", error)
            return
        # Show completed previews during motion, but never show an old CT.
        # The next dispatch always takes the latest settings, not a FIFO queue.
        if state.volume is not self.volume:
            return
        try:
            geometry = state.geometry
            dy = geometry.pixel_spacing_y_mm or geometry.pixel_spacing_mm
            aspect = (geometry.detector_width_px * geometry.pixel_spacing_mm
                      / (geometry.detector_height_px * dy))
            self.raw_image_view.set_image(raw, display_aspect=aspect)
            self.image_view.set_image(display, display_aspect=aspect)
            status = (
                f"{'Preview' if state.preview else 'Rendered'} "
                f"{display.shape[1]}x{display.shape[0]} frame from "
                f"{state.volume.metadata.source or 'synthetic volume'}."
            )
            if self._initializing:
                self._initial_frame_status = status
            else:
                self._set_status(status)
        except Exception as exc:
            self.render_timer.stop()
            self._set_status(f"Display failed: {exc}")
            QMessageBox.critical(self, "Display Error", str(exc))

    def closeEvent(self, event):  # noqa: N802
        self.render_timer.stop()
        self._worker.stop()
        self._worker.join()
        for view in (self.raw_image_view, self.image_view):
            if view.isValid():
                view.cleanup()
        super().closeEvent(event)

    def reset_controls(self) -> None:
        self.geometry_sliders["sdd"].set_value(DEFAULT_GEOMETRY.source_to_detector_mm)
        self.geometry_sliders["sid"].set_value(DEFAULT_GEOMETRY.source_to_isocenter_mm)
        self.detector_inputs["width"].set_value(DEFAULT_GEOMETRY.detector_width_px)
        self.detector_inputs["height"].set_value(DEFAULT_GEOMETRY.detector_height_px)
        self.pixel_spacing_input.set_value(DEFAULT_GEOMETRY.pixel_spacing_mm)
        for key in ("rx", "ry", "rz", "tx", "ty", "tz"):
            self.pose_sliders[key].set_value(0.0)
        self.realism_checkbox.setChecked(False)
        for slider in self.realism_sliders.values():
            slider.set_value(0.0)

    def toggle_realism(self, checked: bool) -> None:
        self.realism_enabled = checked
        self.realism_sliders_group.setEnabled(checked)

    def load_ct_volume(self) -> None:
        path_str, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Image or Volume",
            str(Path.cwd()),
            (
                "Supported Files (*.nii *.nii.gz *.npy *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
                "All Files (*)"
            ),
        )
        if not path_str:
            return

        try:
            path = Path(path_str)
            self.volume = load_ct_volume_from_path(path)
            self._set_status(f"Loaded: {path}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            QMessageBox.critical(self, "Load Error", str(exc))


def check_gpu_dependencies() -> None:
    """Fail before loading a volume when the active venv cannot run the viewer."""
    from importlib.util import find_spec

    try:
        import torch
        import cupy
        # Importing cuCIM before the first Slang CUDA device can crash in
        # rhiCudaDriverApiInit (observed with slangpy 0.43.1 / cuCIM 26.10).
        # Probe only the top-level package: find_spec on a dotted name would
        # import its parents. process_radiograph imports cuCIM after rendering.
        if find_spec("cucim") is None:
            raise ImportError("cuCIM is not installed")
        from cuda.bindings import runtime
        from OpenGL import GL
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"GPU viewer dependencies could not be imported by {sys.executable}: {exc}\n"
            "Install the CUDA extra into this same environment:\n"
            f'  "{sys.executable}" -m pip install -e "{_PROJECT_DIR}[qt-cuda12]"\n'
            "Use qt-cuda13 instead for a CUDA 13 environment.\n"
            "The batch renderer needs CUDA-enabled PyTorch even though image processing uses CuPy."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} in {sys.executable} cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver."
        )


def main() -> int:
    try:
        check_gpu_dependencies()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    surface_format = QSurfaceFormat()
    surface_format.setVersion(3, 3)
    try:
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    except AttributeError:
        surface_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(surface_format)
    app = QApplication(sys.argv)
    window = XraySimulatorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
