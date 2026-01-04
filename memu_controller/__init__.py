"""
BlueStacks Emulator Controller Package

A modular Python package for controlling BlueStacks emulators via ADB.
"""

from memu_controller.controller import BlueStacksController
from memu_controller.config import BlueStacksConfig
from memu_controller.vm_manager import VMManager
from memu_controller.adb_manager import ADBManager
from memu_controller.input_manager import InputManager
from memu_controller.image_utils import ImageProcessor
from memu_controller.coordinate_utils import CoordinateConverter

__version__ = "2.0.0"
__all__ = [
    "BlueStacksController",
    "BlueStacksConfig",
    "VMManager",
    "ADBManager",
    "InputManager",
    "ImageProcessor",
    "CoordinateConverter",
]

