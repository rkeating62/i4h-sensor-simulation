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

    Map/copy/unmap runs on the producing CuPy stream. CUDA's interop barriers
    order the GL texture upload after the device copy without host readback.
    Resources belong to this widget's GL context and are recreated on resize
    of the detector or on Qt context replacement.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self._resource = None
        self._pbo = self._texture = self._program = self._vao = 0
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
        if self._resource is not None:
            _cuda_result(self.cuda.cudaGraphicsUnregisterResource(self._resource))
            self._resource = None
        if self._pbo:
            self.gl.glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
        if self._texture:
            self.gl.glDeleteTextures([self._texture])
            self._texture = 0
        self._shape = None

    def set_image(self, image):
        """Upload device pixels while this widget's OpenGL context is current."""
        if not self.isValid():
            raise RuntimeError("A working NVIDIA OpenGL context is required")
        self.makeCurrent()
        try:
            if self._error is not None:
                raise RuntimeError(f"Cannot initialize CUDA–OpenGL display: {self._error}")
            cp, GL = self.cp, self.gl
            if not isinstance(image, cp.ndarray) or image.ndim != 2 or image.dtype != cp.uint16:
                raise ValueError("Display expects a 2D CuPy uint16 image")
            with image.device:
                image = cp.ascontiguousarray(image)
                height, width = image.shape
                if self._shape != image.shape:
                    self._release_image()
                    self._pbo = GL.glGenBuffers(1)
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo)
                    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, image.nbytes, None, GL.GL_STREAM_DRAW)
                    self._resource, = _cuda_result(self.cuda.cudaGraphicsGLRegisterBuffer(
                        int(self._pbo), self.cuda.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
                    ))
                    self._texture = GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16, width, height,
                                    0, GL.GL_RED, GL.GL_UNSIGNED_SHORT, None)
                    self._shape = image.shape
                stream = cp.cuda.get_current_stream()
                # cuda-python accepts the handle wrapper for a single resource,
                # not a Python list. The wrapper supplies the handle's address.
                _cuda_result(self.cuda.cudaGraphicsMapResources(1, self._resource, stream.ptr))
                try:
                    ptr, size = _cuda_result(self.cuda.cudaGraphicsResourceGetMappedPointer(self._resource))
                    if size < image.nbytes:
                        raise RuntimeError("Mapped OpenGL buffer is smaller than the image")
                    cp.cuda.runtime.memcpyAsync(int(ptr), image.data.ptr, image.nbytes,
                                                cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                finally:
                    _cuda_result(self.cuda.cudaGraphicsUnmapResources(1, self._resource, stream.ptr))
                # Retain the source until the asynchronous device copy completes.
                stream.synchronize()
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                                   GL.GL_RED, GL.GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
                self._image = image
        finally:
            if hasattr(self, "gl"):
                self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)
            self.doneCurrent()
        self.update()

    def paintGL(self):  # noqa: N802
        if self._error is not None:
            return
        GL = self.gl
        GL.glClearColor(0.06, 0.06, 0.06, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._shape is None:
            return
        rows, cols = self._shape
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        scale = min(width / cols, height / rows)
        vw, vh = round(cols * scale), round(rows * scale)
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
    ) -> None:
        super().__init__(parent)
        self.scale = scale

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
        self.value_label.setText(f"{value / self.scale:.2f}")

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


class XraySimulatorWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"X-Ray Simulation Viewer ({QT_API})")
        self.resize(1800, 950)

        self.volume: PreprocessedVolume = load_cached_volume()
        self.realism_enabled = False
        self.simulator: xray_simulator | None = None
        self.current_rotation = self.current_translation = None
        self._needs_simulator_reset = True

        self.render_timer = QTimer(self)
        #self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_current_view)

        self._build_ui()
        self._set_status("Loaded synthetic demo volume.")
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
            "sdd": SliderRow("Source-Detector [mm]", 300, 3000, int(DEFAULT_GEOMETRY.source_to_detector_mm)),
            "sid": SliderRow("Source-Iso [mm]", 250, 2950, int(DEFAULT_GEOMETRY.source_to_isocenter_mm)),
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
           "rx": SliderRow("X Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "ry": SliderRow("Y Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "rz": SliderRow("Z Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "tx": SliderRow("X Translation [mm]", -1500, 1500, 0),
            "ty": SliderRow("Y Translation [mm]", -1500, 1500, 0),
            "tz": SliderRow("Z Translation [mm]", -1500, 1500, 0),
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
            "poisson": SliderRow("Poisson Noise [photons]", 0, 100, 0, scale=0.01),
            "gaussian": SliderRow("Gaussian Noise Sigma", 0, 100, 0, scale=400.0),
            "blur": SliderRow("Gaussian Blur [px]", 0, 100, 0, scale=10.0),
        }
        for slider in self.realism_sliders.values():
            slider.slider.valueChanged.connect(self.on_realism_slider_changed)
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

    def _rebuild_simulator_if_needed(self) -> bool:
        sid = self.geometry_sliders["sid"].value()
        sdd = max(self.geometry_sliders["sdd"].value(), sid + 1.0)

        if sdd != self.geometry_sliders["sdd"].value():
            self.geometry_sliders["sdd"].set_value(sdd)

        new_simulator_required = (
            self.simulator is None
            or self._needs_simulator_reset
            or self.simulator.config.geometry != self.geometry_controls()
            or self.simulator.config.realism != self.realism_controls()
        )
        if not new_simulator_required:
            return False

        config = SimulatorConfig(
            geometry=self.geometry_controls(),
            physics=DEFAULT_PHYSICS,
            realism=self.realism_controls(),
        ).with_output(keep_intensity=True)
        self.simulator = xray_simulator(self.volume, config)
        self._needs_simulator_reset = False
        return True

    def rotation_controls(self) -> tuple[float, float, float]:
        return (
            math.radians(self.pose_sliders["rx"].value() - 90.0),
            math.radians(self.pose_sliders["ry"].value()),
            math.radians(self.pose_sliders["rz"].value()),
        )

    def translation_controls(self) -> tuple[float, float, float]:
        return (
            self.pose_sliders["tx"].value(),
            self.pose_sliders["ty"].value(),
            self.pose_sliders["tz"].value(),
        )

    def render_current_view(self) -> None:
        try:
            simulator_changed = self._rebuild_simulator_if_needed()
            assert self.simulator is not None
            do_render = ( simulator_changed
                         or self.current_rotation != self.rotation_controls()
                         or self.current_translation != self.translation_controls()
            )
            if do_render:
                self.current_rotation = self.rotation_controls()
                self.current_translation = self.translation_controls()
                import cupy as cp

                # Batch size one uses the renderer's CUDA buffer export, avoiding
                # render_frame's NumPy readback and CPU display/realism stages.
                batch = self.simulator.renderer.render_batch(
                    np.asarray([self.current_rotation], dtype=np.float32),
                    np.asarray([self.current_translation], dtype=np.float32),
                    as_torch=True,
                )
                if not batch.is_cuda:
                    raise RuntimeError("The viewer requires the CUDA rendering backend")
                sensor = cp.from_dlpack(batch[0])
                sensor = self._apply_gpu_realism(sensor)
                radiograph = process_radiograph(
                    sensor,
                    i0=self.simulator.config.physics.i0,
                    backend="cuda",
                    equalization="adaptive",
                    median_radius=0,
                    adaptive_radius=16,
                    percentiles=(0.1, 99.9),
                )
                self.raw_image_view.set_image(self._format_raw_for_display(sensor))
                self.image_view.set_image(radiograph.display)
                self._set_status(
                    f"Rendered {sensor.shape[1]}x{sensor.shape[0]} frame from "
                    f"{self.volume.metadata.source or 'synthetic volume'}."
                )
        except Exception as exc:
            self.render_timer.stop()
            self._set_status(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Error", str(exc))

    def _apply_gpu_realism(self, image):
        """Apply the viewer's detector noise and blur controls on the GPU."""
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter

        cfg = self.simulator.config.realism
        if not cfg.enabled:
            return image
        out = cp.maximum(image * cfg.gain + cfg.bias, 0)
        rng = cp.random.RandomState(cfg.seed)
        if cfg.poisson_photons > 0:
            out = rng.poisson(out * cfg.poisson_photons).astype(cp.float32) / cfg.poisson_photons
        if cfg.gaussian_sigma > 0:
            out = cp.maximum(out + rng.normal(0, cfg.gaussian_sigma, out.shape).astype(cp.float32), 0)
        if cfg.blur_sigma_px > 0:
            out = gaussian_filter(out, cfg.blur_sigma_px)
        return out

    def _format_raw_for_display(self, image):
        """Stretch linear intensities to uint16 using device operations only."""
        import cupy as cp

        low, high = image.min(), image.max()
        span = high - low
        normalized = cp.where(span > 0, (image - low) / cp.maximum(span, 1e-20), 0)
        return cp.rint(cp.clip(1 - normalized, 0, 1) * 65535).astype(cp.uint16)

    def closeEvent(self, event):  # noqa: N802
        self.render_timer.stop()
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
        self._needs_simulator_reset = True

    def on_realism_slider_changed(self, _value: int) -> None:
        self._needs_simulator_reset = True

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
            self._needs_simulator_reset = True
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
