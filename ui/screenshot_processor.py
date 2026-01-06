"""
Interactive Screenshot Processor - Main PyQt application.

This module provides a PyQt-based UI for processing BlueStacks VM screenshots
with real-time OpenCV filtering, area selection, OCR operations, and
configuration export.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import cv2

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QScrollArea, QSpinBox, QDoubleSpinBox,
        QComboBox, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
        QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSplitter
    )
    from PyQt6.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
    from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QMouseEvent
    from PIL.ImageQt import ImageQt
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QPushButton, QLabel, QScrollArea, QSpinBox, QDoubleSpinBox,
            QComboBox, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
            QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSplitter
        )
        from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
        from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QMouseEvent
        
        # For PyQt5, PIL.ImageQt doesn't export ImageQt class directly
        # We'll create a compatibility wrapper that converts PIL Image to QImage
        def _pil_to_qimage(pil_image):
            """Convert PIL Image to QImage (PyQt5 compatible)."""
            if pil_image.mode == "RGB":
                qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)
            elif pil_image.mode == "RGBA":
                qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            elif pil_image.mode == "L":
                qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_Grayscale8)
            else:
                # Convert to RGB if mode is not supported
                rgb_image = pil_image.convert("RGB")
                qimage = QImage(rgb_image.tobytes(), rgb_image.width, rgb_image.height, QImage.Format_RGB888)
            return qimage
        
        # Create ImageQt compatibility class for PyQt5
        # It should be callable like ImageQt(image) and return QImage
        class ImageQt:
            def __new__(cls, image):
                return _pil_to_qimage(image)
        
        PYQT_VERSION = 5
    except ImportError:
        raise ImportError("PyQt5 or PyQt6 is required. Install with: pip install PyQt6")

# Ensure project root is in sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from android_controller import BlueStacksController
from android_controller.image_utils import ImageProcessor
from ui.screenshot_processor_utils import (
    FilterProcessor, OCRAnalyzer, CoordinateConverter
)


class ImageDisplayWidget(QWidget):
    """
    Widget for displaying screenshot with zoom/pan and area selection.
    """
    
    area_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, parent=None):
        """Initialize image display widget."""
        super().__init__(parent)
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[QPixmap] = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Area selection
        self.selecting_area = False
        self.start_point: Optional[QPoint] = None
        self.current_rect: Optional[QRect] = None
        self.selected_areas: List[Dict[str, Any]] = []  # List of {name, rect, color}
        self.current_area_name = ""
        
        # Click position for color detection
        self.click_position: Optional[QPoint] = None
        
        # Grab hand (pan) functionality
        self.space_pressed = False
        self.panning = False
        self.pan_start_point: Optional[QPoint] = None
        self.pan_start_offset_x = 0
        self.pan_start_offset_y = 0
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        if PYQT_VERSION == 6:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Enable keyboard focus
        else:
            self.setFocusPolicy(Qt.StrongFocus)  # PyQt5 compatibility
    
    def set_image(self, image: Image.Image):
        """Set the image to display."""
        self.original_image = image
        self._update_display()
    
    def _update_display(self):
        """Update the display pixmap."""
        if self.original_image is None:
            return
        
        # Convert PIL to QPixmap
        qimage = ImageQt(self.original_image)
        self.display_image = QPixmap.fromImage(qimage)
        self.update()
    
    def paintEvent(self, event):
        """Paint the image and overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.display_image is None:
            painter.fillRect(self.rect(), Qt.GlobalColor.lightGray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return
        
        # Calculate display rect
        img_width = self.display_image.width()
        img_height = self.display_image.height()
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        x = self.offset_x + (self.width() - display_width) // 2
        y = self.offset_y + (self.height() - display_height) // 2
        
        # Convert to integers - drawPixmap requires int arguments
        x = int(x)
        y = int(y)
        display_width = int(display_width)
        display_height = int(display_height)
        
        # Draw image
        painter.drawPixmap(x, y, display_width, display_height, self.display_image)
        
        # Draw selected areas
        for area in self.selected_areas:
            rect = area['rect']
            name = area['name']
            color = area['color']
            
            # Convert absolute coordinates to display coordinates
            display_rect = self._absolute_to_display(rect)
            if display_rect:
                pen = QPen(QColor(*color), 2)
                painter.setPen(pen)
                painter.drawRect(display_rect)
                
                # Draw label
                painter.fillRect(display_rect.x(), display_rect.y() - 20, 100, 20, QColor(*color))
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.drawText(display_rect.x() + 5, display_rect.y() - 5, name)
        
        # Draw current selection
        if self.current_rect:
            display_rect = self._absolute_to_display(self.current_rect)
            if display_rect:
                pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(display_rect)
    
    def _absolute_to_display(self, rect: QRect) -> Optional[QRect]:
        """Convert absolute image coordinates to display coordinates."""
        if self.original_image is None or self.display_image is None:
            return None
        
        img_width = self.original_image.width
        img_height = self.original_image.height
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        x_offset = self.offset_x + (self.width() - display_width) // 2
        y_offset = self.offset_y + (self.height() - display_height) // 2
        
        x = int(rect.x() * self.scale_factor) + x_offset
        y = int(rect.y() * self.scale_factor) + y_offset
        w = int(rect.width() * self.scale_factor)
        h = int(rect.height() * self.scale_factor)
        
        # Convert to integers - QRect requires int arguments
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        return QRect(x, y, w, h)
    
    def _display_to_absolute(self, point: QPoint) -> Optional[QPoint]:
        """Convert display coordinates to absolute image coordinates."""
        if self.original_image is None or self.display_image is None:
            return None
        
        img_width = self.original_image.width
        img_height = self.original_image.height
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        x_offset = self.offset_x + (self.width() - display_width) // 2
        y_offset = self.offset_y + (self.height() - display_height) // 2
        
        x = int((point.x() - x_offset) / self.scale_factor)
        y = int((point.y() - y_offset) / self.scale_factor)
        
        if 0 <= x < img_width and 0 <= y < img_height:
            return QPoint(x, y)
        return None
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for area selection or panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Handle both PyQt5 and PyQt6
            if PYQT_VERSION == 6:
                pos = event.position().toPoint()
            else:
                pos = event.pos()
            
            # Check if space is pressed for grab hand mode
            if self.space_pressed:
                # Start panning
                self.panning = True
                self.pan_start_point = pos
                self.pan_start_offset_x = self.offset_x
                self.pan_start_offset_y = self.offset_y
                # Change cursor to closed hand (grab)
                if PYQT_VERSION == 6:
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                else:
                    self.setCursor(Qt.ClosedHandCursor)
            else:
                # Normal area selection
                abs_point = self._display_to_absolute(pos)
                if abs_point:
                    self.selecting_area = True
                    self.start_point = abs_point
                    self.current_rect = QRect(abs_point, QSize(0, 0))
                    self.click_position = abs_point
                    self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for area selection or panning."""
        # Handle both PyQt5 and PyQt6
        if PYQT_VERSION == 6:
            pos = event.position().toPoint()
        else:
            pos = event.pos()
        
        if self.panning and self.pan_start_point is not None:
            # Pan the image
            dx = pos.x() - self.pan_start_point.x()
            dy = pos.y() - self.pan_start_point.y()
            self.offset_x = self.pan_start_offset_x + dx
            self.offset_y = self.pan_start_offset_y + dy
            self.update()
        elif self.selecting_area and self.start_point:
            # Area selection
            abs_point = self._display_to_absolute(pos)
            if abs_point:
                self.current_rect = QRect(
                    min(self.start_point.x(), abs_point.x()),
                    min(self.start_point.y(), abs_point.y()),
                    abs(abs_point.x() - self.start_point.x()),
                    abs(abs_point.y() - self.start_point.y())
                )
                self.update()
        elif self.space_pressed:
            # Show open hand cursor when space is pressed but not dragging yet
            if PYQT_VERSION == 6:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to complete area selection or panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.panning:
                # End panning
                self.panning = False
                self.pan_start_point = None
                if self.space_pressed:
                    self.setCursor(Qt.OpenHandCursor if PYQT_VERSION == 5 else Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor if PYQT_VERSION == 5 else Qt.CursorShape.ArrowCursor)
            elif self.selecting_area:
                # Complete area selection
                self.selecting_area = False
                if self.current_rect and self.current_rect.width() > 5 and self.current_rect.height() > 5:
                    self.area_selected.emit(
                        self.current_rect.x(),
                        self.current_rect.y(),
                        self.current_rect.width(),
                        self.current_rect.height()
                    )
                self.current_rect = None
                self.update()
    
    def add_area(self, name: str, x: int, y: int, width: int, height: int, color: Tuple[int, int, int] = (0, 255, 0)):
        """Add a named area to the display."""
        rect = QRect(x, y, width, height)
        self.selected_areas.append({
            'name': name,
            'rect': rect,
            'color': color
        })
        self.update()
    
    def clear_areas(self):
        """Clear all selected areas."""
        self.selected_areas.clear()
        self.update()
    
    def remove_area(self, name: str):
        """Remove an area by name."""
        self.selected_areas = [a for a in self.selected_areas if a['name'] != name]
        self.update()
    
    def get_click_position(self) -> Optional[QPoint]:
        """Get the last clicked position."""
        return self.click_position
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in on the image."""
        old_scale = self.scale_factor
        self.scale_factor *= factor
        self.scale_factor = min(10.0, self.scale_factor)  # Max zoom 10x
        
        # Adjust offset to zoom towards center
        if self.original_image:
            center_x = self.width() // 2
            center_y = self.height() // 2
            # Calculate how much the image grew
            scale_diff = self.scale_factor - old_scale
            img_width = self.original_image.width
            img_height = self.original_image.height
            self.offset_x -= (center_x - self.width() // 2) * scale_diff / old_scale
            self.offset_y -= (center_y - self.height() // 2) * scale_diff / old_scale
        
        self.update()
        # Notify parent if callback exists
        if hasattr(self, 'zoom_changed'):
            self.zoom_changed()
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out from the image."""
        old_scale = self.scale_factor
        self.scale_factor /= factor
        self.scale_factor = max(0.1, self.scale_factor)  # Min zoom 0.1x
        
        # Adjust offset to zoom towards center
        if self.original_image:
            center_x = self.width() // 2
            center_y = self.height() // 2
            # Calculate how much the image shrank
            scale_diff = old_scale - self.scale_factor
            img_width = self.original_image.width
            img_height = self.original_image.height
            self.offset_x += (center_x - self.width() // 2) * scale_diff / old_scale
            self.offset_y += (center_y - self.height() // 2) * scale_diff / old_scale
        
        self.update()
        # Notify parent if callback exists
        if hasattr(self, 'zoom_changed'):
            self.zoom_changed()
    
    def zoom_fit(self):
        """Zoom to fit the image in the widget."""
        if self.original_image is None or self.display_image is None:
            return
        
        img_width = self.original_image.width
        img_height = self.original_image.height
        widget_width = self.width()
        widget_height = self.height()
        
        # Calculate scale to fit
        scale_x = widget_width / img_width if img_width > 0 else 1.0
        scale_y = widget_height / img_height if img_height > 0 else 1.0
        self.scale_factor = min(scale_x, scale_y) * 0.95  # 95% to add some padding
        
        # Reset offset
        self.offset_x = 0
        self.offset_y = 0
        
        self.update()
        # Notify parent if callback exists
        if hasattr(self, 'zoom_changed'):
            self.zoom_changed()
    
    def zoom_reset(self):
        """Reset zoom to 100%."""
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update()
        # Notify parent if callback exists
        if hasattr(self, 'zoom_changed'):
            self.zoom_changed()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == (Qt.Key.Key_Space if PYQT_VERSION == 6 else Qt.Key_Space):
            self.space_pressed = True
            if not self.panning:
                self.setCursor(Qt.OpenHandCursor if PYQT_VERSION == 5 else Qt.CursorShape.OpenHandCursor)
            event.accept()
        elif event.key() in ((Qt.Key.Key_Plus, Qt.Key.Key_Equal) if PYQT_VERSION == 6 else (Qt.Key_Plus, Qt.Key_Equal)):
            # Zoom in with + or =
            self.zoom_in()
            event.accept()
        elif event.key() == (Qt.Key.Key_Minus if PYQT_VERSION == 6 else Qt.Key_Minus):
            # Zoom out with -
            self.zoom_out()
            event.accept()
        elif event.key() == (Qt.Key.Key_0 if PYQT_VERSION == 6 else Qt.Key_0):
            # Reset zoom with 0
            self.zoom_reset()
            event.accept()
        elif event.key() == (Qt.Key.Key_F if PYQT_VERSION == 6 else Qt.Key_F):
            # Fit to window with F
            self.zoom_fit()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.key() == (Qt.Key.Key_Space if PYQT_VERSION == 6 else Qt.Key_Space):
            self.space_pressed = False
            if not self.panning:
                self.setCursor(Qt.ArrowCursor if PYQT_VERSION == 5 else Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    def enterEvent(self, event):
        """Handle mouse enter event."""
        self.setFocus()  # Get keyboard focus when mouse enters
        super().enterEvent(event)


class ScreenshotProcessorWindow(QMainWindow):
    """
    Main window for the Interactive Screenshot Processor.
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Interactive Screenshot Processor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Image data
        self.current_image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.processed_image: Optional[np.ndarray] = None
        
        # Controller
        self.controller: Optional[BlueStacksController] = None
        self.vm_index = 0
        
        # OCR analyzer
        self.ocr_analyzer = OCRAnalyzer()
        
        # Areas storage (percentage-based)
        self.areas: Dict[str, Dict[str, float]] = {}
        
        # Filter parameters
        self.filter_params = {
            'threshold': {'enabled': False, 'type': 'binary', 'value': 127},
            'blur': {'enabled': False, 'type': 'gaussian', 'kernel_size': 5, 'sigma': 1.0},
            'morphology': {'enabled': False, 'operation': 'open', 'kernel_size': 3, 'shape': 'rect', 'iterations': 1},
            'edge_detection': {'enabled': False, 'type': 'canny', 'threshold1': 50, 'threshold2': 150},
            'color_space': {'enabled': False, 'type': 'grayscale'},
            'contour': {'enabled': False, 'min_area': 100, 'max_area': 1000000},
            'hough_lines': {'enabled': False, 'threshold': 100, 'min_line_length': 50},
            'hough_circles': {'enabled': False, 'param1': 50, 'param2': 30}
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Image display (full height)
        self.image_display = ImageDisplayWidget()
        self.image_display.area_selected.connect(self._on_area_selected)
        # Store reference to update zoom label
        self.image_display.zoom_changed = lambda: self._update_zoom_label()
        
        splitter.addWidget(self.image_display)
        
        # Right panel: Controls
        right_panel = self._create_controls_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    
    def _create_controls_panel(self) -> QWidget:
        """Create the controls panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Image loading and zoom controls at the top
        controls_group = self._create_image_controls_panel()
        layout.addWidget(controls_group)
        
        # Scroll area for controls
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Filters panel
        filters_group = self._create_filters_panel()
        scroll_layout.addWidget(filters_group)
        
        # Area management panel
        area_group = self._create_area_panel()
        scroll_layout.addWidget(area_group)
        
        # OCR operations panel
        ocr_group = self._create_ocr_panel()
        scroll_layout.addWidget(ocr_group)
        
        # Results panel
        results_group = self._create_results_panel()
        scroll_layout.addWidget(results_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        return panel
    
    def _create_image_controls_panel(self) -> QGroupBox:
        """Create the image loading and zoom controls panel."""
        group = QGroupBox("Image Controls")
        layout = QVBoxLayout()
        
        # Image loading buttons
        load_layout = QHBoxLayout()
        self.btn_load_vm = QPushButton("Load from VM")
        self.btn_load_vm.clicked.connect(self._load_from_vm)
        load_layout.addWidget(self.btn_load_vm)
        
        self.btn_load_file = QPushButton("Load from File")
        self.btn_load_file.clicked.connect(self._load_from_file)
        load_layout.addWidget(self.btn_load_file)
        
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_screenshot)
        load_layout.addWidget(self.btn_refresh)
        
        layout.addLayout(load_layout)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_layout.addWidget(zoom_label)
        
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setToolTip("Zoom In (or use + key)")
        self.btn_zoom_in.setMaximumWidth(30)
        self.btn_zoom_in.clicked.connect(lambda: self.image_display.zoom_in())
        zoom_layout.addWidget(self.btn_zoom_in)
        
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setToolTip("Zoom Out (or use - key)")
        self.btn_zoom_out.setMaximumWidth(30)
        self.btn_zoom_out.clicked.connect(lambda: self.image_display.zoom_out())
        zoom_layout.addWidget(self.btn_zoom_out)
        
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setToolTip("Fit to Window (or use F key)")
        self.btn_zoom_fit.setMaximumWidth(40)
        self.btn_zoom_fit.clicked.connect(lambda: self.image_display.zoom_fit())
        zoom_layout.addWidget(self.btn_zoom_fit)
        
        self.btn_zoom_reset = QPushButton("100%")
        self.btn_zoom_reset.setToolTip("Reset Zoom to 100% (or use 0 key)")
        self.btn_zoom_reset.setMaximumWidth(50)
        self.btn_zoom_reset.clicked.connect(lambda: self.image_display.zoom_reset())
        zoom_layout.addWidget(self.btn_zoom_reset)
        
        # Zoom level display
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        zoom_layout.addWidget(self.zoom_label)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Grab hand hint
        grab_label = QLabel("Hold Space + Drag to pan")
        grab_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(grab_label)
        
        # Export button
        self.btn_export = QPushButton("Export Config")
        self.btn_export.clicked.connect(self._export_config)
        layout.addWidget(self.btn_export)
        
        group.setLayout(layout)
        return group
    
    def _create_filters_panel(self) -> QGroupBox:
        """Create the filters control panel."""
        group = QGroupBox("Filters")
        layout = QVBoxLayout()
        
        # Threshold
        threshold_group = QGroupBox("Threshold")
        threshold_layout = QVBoxLayout()
        
        self.threshold_enable = QCheckBox("Enable")
        threshold_layout.addWidget(self.threshold_enable)
        
        threshold_type_layout = QHBoxLayout()
        threshold_type_layout.addWidget(QLabel("Type:"))
        self.threshold_type = QComboBox()
        self.threshold_type.addItems(["binary", "binary_inv", "adaptive", "otsu"])
        threshold_type_layout.addWidget(self.threshold_type)
        threshold_layout.addLayout(threshold_type_layout)
        
        threshold_value_layout = QHBoxLayout()
        threshold_value_layout.addWidget(QLabel("Value:"))
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(127)
        threshold_value_layout.addWidget(self.threshold_value)
        threshold_layout.addLayout(threshold_value_layout)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Blur
        blur_group = QGroupBox("Blur")
        blur_layout = QVBoxLayout()
        
        self.blur_enable = QCheckBox("Enable")
        blur_layout.addWidget(self.blur_enable)
        
        blur_type_layout = QHBoxLayout()
        blur_type_layout.addWidget(QLabel("Type:"))
        self.blur_type = QComboBox()
        self.blur_type.addItems(["gaussian", "median", "bilateral"])
        blur_type_layout.addWidget(self.blur_type)
        blur_layout.addLayout(blur_type_layout)
        
        blur_kernel_layout = QHBoxLayout()
        blur_kernel_layout.addWidget(QLabel("Kernel:"))
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(3, 51)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(5)
        blur_kernel_layout.addWidget(self.blur_kernel)
        blur_layout.addLayout(blur_kernel_layout)
        
        blur_sigma_layout = QHBoxLayout()
        blur_sigma_layout.addWidget(QLabel("Sigma:"))
        self.blur_sigma = QDoubleSpinBox()
        self.blur_sigma.setRange(0.1, 10.0)
        self.blur_sigma.setSingleStep(0.1)
        self.blur_sigma.setValue(1.0)
        blur_sigma_layout.addWidget(self.blur_sigma)
        blur_layout.addLayout(blur_sigma_layout)
        
        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)
        
        # Morphology
        morph_group = QGroupBox("Morphology")
        morph_layout = QVBoxLayout()
        
        self.morph_enable = QCheckBox("Enable")
        morph_layout.addWidget(self.morph_enable)
        
        morph_op_layout = QHBoxLayout()
        morph_op_layout.addWidget(QLabel("Operation:"))
        self.morph_op = QComboBox()
        self.morph_op.addItems(["erode", "dilate", "open", "close", "gradient"])
        morph_op_layout.addWidget(self.morph_op)
        morph_layout.addLayout(morph_op_layout)
        
        morph_kernel_layout = QHBoxLayout()
        morph_kernel_layout.addWidget(QLabel("Kernel:"))
        self.morph_kernel = QSpinBox()
        self.morph_kernel.setRange(3, 21)
        self.morph_kernel.setValue(3)
        morph_kernel_layout.addWidget(self.morph_kernel)
        morph_layout.addLayout(morph_kernel_layout)
        
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        # Edge Detection
        edge_group = QGroupBox("Edge Detection")
        edge_layout = QVBoxLayout()
        
        self.edge_enable = QCheckBox("Enable")
        edge_layout.addWidget(self.edge_enable)
        
        edge_type_layout = QHBoxLayout()
        edge_type_layout.addWidget(QLabel("Type:"))
        self.edge_type = QComboBox()
        self.edge_type.addItems(["canny", "sobel_x", "sobel_y", "sobel_xy", "laplacian"])
        edge_type_layout.addWidget(self.edge_type)
        edge_layout.addLayout(edge_type_layout)
        
        edge_thresh1_layout = QHBoxLayout()
        edge_thresh1_layout.addWidget(QLabel("Thresh1:"))
        self.edge_thresh1 = QSpinBox()
        self.edge_thresh1.setRange(0, 255)
        self.edge_thresh1.setValue(50)
        edge_thresh1_layout.addWidget(self.edge_thresh1)
        edge_layout.addLayout(edge_thresh1_layout)
        
        edge_thresh2_layout = QHBoxLayout()
        edge_thresh2_layout.addWidget(QLabel("Thresh2:"))
        self.edge_thresh2 = QSpinBox()
        self.edge_thresh2.setRange(0, 255)
        self.edge_thresh2.setValue(150)
        edge_thresh2_layout.addWidget(self.edge_thresh2)
        edge_layout.addLayout(edge_thresh2_layout)
        
        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)
        
        # Color Space
        color_group = QGroupBox("Color Space")
        color_layout = QVBoxLayout()
        
        self.color_enable = QCheckBox("Enable")
        color_layout.addWidget(self.color_enable)
        
        color_type_layout = QHBoxLayout()
        color_type_layout.addWidget(QLabel("Type:"))
        self.color_type = QComboBox()
        self.color_type.addItems(["grayscale", "hsv", "lab", "rgb"])
        color_type_layout.addWidget(self.color_type)
        color_layout.addLayout(color_type_layout)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Connect signals
        self.threshold_enable.toggled.connect(self._apply_filters)
        self.threshold_type.currentTextChanged.connect(self._apply_filters)
        self.threshold_value.valueChanged.connect(self._apply_filters)
        
        self.blur_enable.toggled.connect(self._apply_filters)
        self.blur_type.currentTextChanged.connect(self._apply_filters)
        self.blur_kernel.valueChanged.connect(self._apply_filters)
        self.blur_sigma.valueChanged.connect(self._apply_filters)
        
        self.morph_enable.toggled.connect(self._apply_filters)
        self.morph_op.currentTextChanged.connect(self._apply_filters)
        self.morph_kernel.valueChanged.connect(self._apply_filters)
        
        self.edge_enable.toggled.connect(self._apply_filters)
        self.edge_type.currentTextChanged.connect(self._apply_filters)
        self.edge_thresh1.valueChanged.connect(self._apply_filters)
        self.edge_thresh2.valueChanged.connect(self._apply_filters)
        
        self.color_enable.toggled.connect(self._apply_filters)
        self.color_type.currentTextChanged.connect(self._apply_filters)
        
        group.setLayout(layout)
        return group
    
    def _create_area_panel(self) -> QGroupBox:
        """Create the area management panel."""
        group = QGroupBox("Area Management")
        layout = QVBoxLayout()
        
        # Area name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Area Name:"))
        self.area_name_input = QLineEdit()
        self.area_name_input.setPlaceholderText("e.g., grid_region")
        name_layout.addWidget(self.area_name_input)
        layout.addLayout(name_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_area = QPushButton("Add Area")
        self.btn_add_area.clicked.connect(self._add_area_manual)
        btn_layout.addWidget(self.btn_add_area)
        
        self.btn_delete_area = QPushButton("Delete")
        self.btn_delete_area.clicked.connect(self._delete_area)
        btn_layout.addWidget(self.btn_delete_area)
        
        self.btn_clear_areas = QPushButton("Clear All")
        self.btn_clear_areas.clicked.connect(self._clear_areas)
        btn_layout.addWidget(self.btn_clear_areas)
        
        layout.addLayout(btn_layout)
        
        # Areas list
        layout.addWidget(QLabel("Selected Areas:"))
        self.areas_list = QListWidget()
        layout.addWidget(self.areas_list)
        
        group.setLayout(layout)
        return group
    
    def _create_ocr_panel(self) -> QGroupBox:
        """Create the OCR operations panel."""
        group = QGroupBox("OCR Operations")
        layout = QVBoxLayout()
        
        self.btn_detect_digits = QPushButton("Detect Digits")
        self.btn_detect_digits.clicked.connect(self._detect_digits)
        layout.addWidget(self.btn_detect_digits)
        
        self.btn_detect_grid = QPushButton("Detect Grid")
        self.btn_detect_grid.clicked.connect(self._detect_grid)
        layout.addWidget(self.btn_detect_grid)
        
        self.btn_detect_color = QPushButton("Detect Color")
        self.btn_detect_color.clicked.connect(self._detect_color)
        layout.addWidget(self.btn_detect_color)
        
        self.btn_detect_lines = QPushButton("Detect Lines")
        self.btn_detect_lines.clicked.connect(self._detect_lines)
        layout.addWidget(self.btn_detect_lines)
        
        self.btn_detect_contours = QPushButton("Detect Contours")
        self.btn_detect_contours.clicked.connect(self._detect_contours)
        layout.addWidget(self.btn_detect_contours)
        
        self.btn_run_all = QPushButton("Run All Operations")
        self.btn_run_all.clicked.connect(self._run_all_operations)
        layout.addWidget(self.btn_run_all)
        
        group.setLayout(layout)
        return group
    
    def _create_results_panel(self) -> QGroupBox:
        """Create the results display panel."""
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        layout.addWidget(self.results_text)
        
        group.setLayout(layout)
        return group
    
    def _load_from_vm(self):
        """Load screenshot from BlueStacks VM."""
        try:
            if self.controller is None:
                self.controller = BlueStacksController()
            
            if not self.controller.connect_adb(self.vm_index):
                QMessageBox.warning(self, "Error", "Failed to connect to VM. Make sure BlueStacks is running.")
                return
            
            screenshot = self.controller.take_screenshot_image(self.vm_index, refresh_display=False)
            if screenshot:
                self.current_image = screenshot
                self.original_image = screenshot.copy()
                self.image_display.set_image(screenshot)
                self._update_zoom_label()
                self._log_result("Screenshot loaded from VM successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to capture screenshot from VM.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading from VM: {str(e)}")
    
    def _load_from_file(self):
        """Load screenshot from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Screenshot",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                self.current_image = image
                self.original_image = image.copy()
                self.image_display.set_image(image)
                self._update_zoom_label()
                self._log_result(f"Screenshot loaded from file: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def _refresh_screenshot(self):
        """Refresh screenshot from VM."""
        self._load_from_vm()
    
    def _on_area_selected(self, x: int, y: int, width: int, height: int):
        """Handle area selection from image display."""
        if not self.current_image:
            return
        
        name = self.area_name_input.text().strip()
        if not name:
            name = f"area_{len(self.areas) + 1}"
        
        # Convert to percentage
        img_width = self.current_image.width
        img_height = self.current_image.height
        x_pct, y_pct, width_pct, height_pct = CoordinateConverter.absolute_to_percentage(
            x, y, width, height, img_width, img_height
        )
        
        # Store area
        self.areas[name] = {
            'x_pct': x_pct,
            'y_pct': y_pct,
            'width_pct': width_pct,
            'height_pct': height_pct
        }
        
        # Add to display
        import random
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.image_display.add_area(name, x, y, width, height, color)
        
        # Update list
        self._update_areas_list()
        
        self._log_result(f"Area '{name}' added: ({x_pct:.2%}, {y_pct:.2%}, {width_pct:.2%}, {height_pct:.2%})")
    
    def _add_area_manual(self):
        """Manually add area (placeholder for future enhancement)."""
        QMessageBox.information(self, "Info", "Select an area by clicking and dragging on the image.")
    
    def _delete_area(self):
        """Delete selected area."""
        current_item = self.areas_list.currentItem()
        if current_item:
            name = current_item.text().split('(')[0].strip()
            if name in self.areas:
                del self.areas[name]
                self.image_display.remove_area(name)
                self._update_areas_list()
                self._log_result(f"Area '{name}' deleted.")
    
    def _clear_areas(self):
        """Clear all areas."""
        self.areas.clear()
        self.image_display.clear_areas()
        self._update_areas_list()
        self._log_result("All areas cleared.")
    
    def _update_areas_list(self):
        """Update the areas list widget."""
        self.areas_list.clear()
        for name, area in self.areas.items():
            text = f"{name} ({area['x_pct']:.1%}, {area['y_pct']:.1%}, {area['width_pct']:.1%}, {area['height_pct']:.1%})"
            self.areas_list.addItem(text)
    
    def _apply_filters(self):
        """Apply all enabled filters to the image."""
        if self.original_image is None:
            return
        
        img_cv = ImageProcessor.pil_to_cv2(self.original_image)
        processed = img_cv.copy()
        
        # Apply filters in order
        if self.threshold_enable.isChecked():
            processed = FilterProcessor.apply_threshold(
                processed,
                self.threshold_type.currentText(),
                self.threshold_value.value()
            )
            # Convert back to BGR if needed
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        if self.blur_enable.isChecked():
            processed = FilterProcessor.apply_blur(
                processed,
                self.blur_type.currentText(),
                self.blur_kernel.value(),
                self.blur_sigma.value()
            )
        
        if self.morph_enable.isChecked():
            processed = FilterProcessor.apply_morphology(
                processed,
                self.morph_op.currentText(),
                self.morph_kernel.value()
            )
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        if self.edge_enable.isChecked():
            processed = FilterProcessor.apply_edge_detection(
                processed,
                self.edge_type.currentText(),
                self.edge_thresh1.value(),
                self.edge_thresh2.value()
            )
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        if self.color_enable.isChecked():
            processed = FilterProcessor.apply_color_space(
                processed,
                self.color_type.currentText()
            )
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Convert back to PIL and update display
        processed_pil = ImageProcessor.cv2_to_pil(processed)
        self.current_image = processed_pil
        self.processed_image = processed
        self.image_display.set_image(processed_pil)
    
    def _detect_digits(self):
        """Detect digits in the current image."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Get selected area if any
        current_item = self.areas_list.currentItem()
        region = None
        if current_item:
            name = current_item.text().split('(')[0].strip()
            if name in self.areas:
                area = self.areas[name]
                img_width = self.current_image.width
                img_height = self.current_image.height
                x, y, w, h = CoordinateConverter.percentage_to_absolute(
                    area['x_pct'], area['y_pct'], area['width_pct'], area['height_pct'],
                    img_width, img_height
                )
                region = (x, y, w, h)
        
        results = self.ocr_analyzer.detect_digits(self.current_image, region)
        
        if results:
            result_text = f"Detected {len(results)} digit(s):\n"
            for r in results:
                result_text += f"  - '{r['text']}' at {r['position']} (confidence: {r['confidence']:.2f})\n"
            self._log_result(result_text)
        else:
            self._log_result("No digits detected.")
    
    def _detect_grid(self):
        """Detect grid structure in the current image."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Get selected area if any
        current_item = self.areas_list.currentItem()
        region = None
        if current_item:
            name = current_item.text().split('(')[0].strip()
            if name in self.areas:
                area = self.areas[name]
                img_width = self.current_image.width
                img_height = self.current_image.height
                x, y, w, h = CoordinateConverter.percentage_to_absolute(
                    area['x_pct'], area['y_pct'], area['width_pct'], area['height_pct'],
                    img_width, img_height
                )
                region = (x, y, w, h)
        
        results = self.ocr_analyzer.detect_grid(self.current_image, region)
        
        result_text = f"Grid Detection Results:\n"
        result_text += f"  - Rows: {results['rows']}\n"
        result_text += f"  - Columns: {results['cols']}\n"
        result_text += f"  - Total Cells: {results['cells']}\n"
        result_text += f"  - Horizontal Lines: {len(results['horizontal_lines'])}\n"
        result_text += f"  - Vertical Lines: {len(results['vertical_lines'])}\n"
        self._log_result(result_text)
    
    def _detect_color(self):
        """Detect color at clicked position or selected area."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Try to get click position first
        click_pos = self.image_display.get_click_position()
        position = None
        if click_pos:
            position = (click_pos.x(), click_pos.y())
        
        # Get selected area if no click position
        region = None
        if not position:
            current_item = self.areas_list.currentItem()
            if current_item:
                name = current_item.text().split('(')[0].strip()
                if name in self.areas:
                    area = self.areas[name]
                    img_width = self.current_image.width
                    img_height = self.current_image.height
                    x, y, w, h = CoordinateConverter.percentage_to_absolute(
                        area['x_pct'], area['y_pct'], area['width_pct'], area['height_pct'],
                        img_width, img_height
                    )
                    region = (x, y, w, h)
        
        results = self.ocr_analyzer.detect_color(self.current_image, position, region)
        
        if 'error' not in results:
            result_text = f"Color Detection:\n"
            result_text += f"  - Type: {results['type']}\n"
            if 'position' in results:
                result_text += f"  - Position: {results['position']}\n"
            if 'region' in results:
                result_text += f"  - Region: {results['region']}\n"
            result_text += f"  - RGB: {results['rgb']}\n"
            result_text += f"  - Hex: {results['hex']}\n"
            self._log_result(result_text)
        else:
            self._log_result("Click on the image or select an area to detect color.")
    
    def _detect_lines(self):
        """Detect lines using Hough transform."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        img_cv = ImageProcessor.pil_to_cv2(self.current_image)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        result_img, lines = FilterProcessor.apply_hough_lines(gray)
        
        result_text = f"Line Detection:\n"
        result_text += f"  - Detected {len(lines)} line(s)\n"
        for i, line in enumerate(lines[:10]):  # Show first 10
            result_text += f"  - Line {i+1}: {line}\n"
        if len(lines) > 10:
            result_text += f"  ... and {len(lines) - 10} more\n"
        
        self._log_result(result_text)
        
        # Update display with lines
        result_pil = ImageProcessor.cv2_to_pil(result_img)
        self.image_display.set_image(result_pil)
    
    def _detect_contours(self):
        """Detect contours in the image."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        img_cv = ImageProcessor.pil_to_cv2(self.current_image)
        result_img, contours = FilterProcessor.apply_contour_detection(img_cv)
        
        result_text = f"Contour Detection:\n"
        result_text += f"  - Detected {len(contours)} contour(s)\n"
        for i, contour in enumerate(contours[:10]):  # Show first 10
            area = cv2.contourArea(contour)
            result_text += f"  - Contour {i+1}: area={area:.0f}\n"
        if len(contours) > 10:
            result_text += f"  ... and {len(contours) - 10} more\n"
        
        self._log_result(result_text)
        
        # Update display with contours
        result_pil = ImageProcessor.cv2_to_pil(result_img)
        self.image_display.set_image(result_pil)
    
    def _run_all_operations(self):
        """Run all OCR operations."""
        self._log_result("=== Running All Operations ===\n")
        self._detect_digits()
        self._detect_grid()
        self._detect_color()
        self._detect_lines()
        self._detect_contours()
        self._log_result("\n=== All Operations Complete ===")
    
    def _update_zoom_label(self):
        """Update the zoom level label in the controls panel."""
        if hasattr(self, 'zoom_label') and self.image_display:
            zoom_percent = int(self.image_display.scale_factor * 100)
            self.zoom_label.setText(f"{zoom_percent}%")
    
    def _log_result(self, text: str):
        """Log result to results panel."""
        self.results_text.append(text)
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _export_config(self):
        """Export configuration as Python code."""
        if not self.areas:
            QMessageBox.warning(self, "Warning", "No areas defined. Add at least one area before exporting.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            "screenshot_config.py",
            "Python Files (*.py)"
        )
        
        if file_path:
            try:
                config_code = self._generate_config_code()
                with open(file_path, 'w') as f:
                    f.write(config_code)
                QMessageBox.information(self, "Success", f"Configuration exported to {file_path}")
                self._log_result(f"Configuration exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting configuration: {str(e)}")
    
    def _generate_config_code(self) -> str:
        """Generate Python configuration code."""
        code = '''# Auto-generated screenshot processing configuration
# Generated by Interactive Screenshot Processor

SCREENSHOT_CONFIG = {
    "filters": {
'''
        
        # Add filter configurations
        filters = []
        if self.threshold_enable.isChecked():
            filters.append(f'        "threshold": {{"type": "{self.threshold_type.currentText()}", "value": {self.threshold_value.value()}}},')
        if self.blur_enable.isChecked():
            filters.append(f'        "blur": {{"type": "{self.blur_type.currentText()}", "kernel_size": {self.blur_kernel.value()}, "sigma": {self.blur_sigma.value()}}},')
        if self.morph_enable.isChecked():
            filters.append(f'        "morphology": {{"operation": "{self.morph_op.currentText()}", "kernel_size": {self.morph_kernel.value()}}},')
        if self.edge_enable.isChecked():
            filters.append(f'        "edge_detection": {{"type": "{self.edge_type.currentText()}", "threshold1": {self.edge_thresh1.value()}, "threshold2": {self.edge_thresh2.value()}}},')
        if self.color_enable.isChecked():
            filters.append(f'        "color_space": {{"type": "{self.color_type.currentText()}}},')
        
        code += '\n'.join(filters) if filters else '        # No filters enabled\n'
        
        code += '''    },
    "areas": {
'''
        
        # Add areas
        area_lines = []
        for name, area in self.areas.items():
            area_lines.append(f'        "{name}": {{')
            area_lines.append(f'            "x_pct": {area["x_pct"]:.6f},')
            area_lines.append(f'            "y_pct": {area["y_pct"]:.6f},')
            area_lines.append(f'            "width_pct": {area["width_pct"]:.6f},')
            area_lines.append(f'            "height_pct": {area["height_pct"]:.6f}')
            area_lines.append(f'        }},')
        
        code += '\n'.join(area_lines)
        code = code.rstrip(',')  # Remove trailing comma
        
        code += '''
    }
}


def get_area_coordinates(image_width, image_height, area_name):
    """
    Get absolute coordinates for an area based on image size.
    
    Parameters
    ----------
    image_width : int
        Width of the image.
    image_height : int
        Height of the image.
    area_name : str
        Name of the area to get coordinates for.
    
    Returns
    -------
    tuple
        Tuple of (x, y, width, height) in absolute coordinates.
    """
    if area_name not in SCREENSHOT_CONFIG["areas"]:
        raise ValueError(f"Area '{area_name}' not found in configuration")
    
    area = SCREENSHOT_CONFIG["areas"][area_name]
    x = int(area["x_pct"] * image_width)
    y = int(area["y_pct"] * image_height)
    width = int(area["width_pct"] * image_width)
    height = int(area["height_pct"] * image_height)
    return (x, y, width, height)


# Example usage:
# from PIL import Image
# image = Image.open("screenshot.png")
# x, y, w, h = get_area_coordinates(image.width, image.height, "grid_region")
# cropped = image.crop((x, y, x + w, y + h))
'''
        
        return code


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = ScreenshotProcessorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

