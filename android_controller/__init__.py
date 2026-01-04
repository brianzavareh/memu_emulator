"""
BlueStacks Emulator Controller Package

A modular Python package for controlling BlueStacks emulators via ADB.
"""

from android_controller.controller import BlueStacksController
from android_controller.config import BlueStacksConfig
from android_controller.vm_manager import VMManager
from android_controller.adb_manager import ADBManager
from android_controller.input_manager import InputManager
from android_controller.image_utils import ImageProcessor
from android_controller.coordinate_utils import CoordinateConverter

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

