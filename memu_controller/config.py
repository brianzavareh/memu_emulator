"""
Configuration management for BlueStacks controller.

This module handles configuration settings, including paths to BlueStacks executables
and default ADB port settings.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BlueStacksConfig:
    """
    Configuration class for BlueStacks controller settings.

    Attributes
    ----------
    adb_path : Optional[str]
        Path to adb.exe. If None, attempts to auto-detect.
    timeout : int
        Default timeout for operations in seconds.
    adb_port_base : int
        Base port for ADB connections. Default is 5555 for BlueStacks.
    """

    adb_path: Optional[str] = None
    timeout: int = 60
    adb_port_base: int = 5555

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BlueStacksConfig":
        """
        Create a BlueStacksConfig instance from a dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration values.

        Returns
        -------
        BlueStacksConfig
            Configured BlueStacksConfig instance.
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "adb_path": self.adb_path,
            "timeout": self.timeout,
            "adb_port_base": self.adb_port_base,
        }

    def get_adb_port(self, instance_index: int = 0) -> int:
        """
        Calculate ADB port for a given BlueStacks instance.

        Parameters
        ----------
        instance_index : int, optional
            Index of the BlueStacks instance. Default is 0.
            For BlueStacks, the first instance typically uses port 5555.

        Returns
        -------
        int
            ADB port number for the instance.
        """
        # BlueStacks typically uses 5555 for the first instance
        # Additional instances may use different ports, but we'll use base + index
        if instance_index == 0:
            return self.adb_port_base
        return self.adb_port_base + instance_index

    @staticmethod
    def find_adb_path() -> Optional[str]:
        """
        Auto-detect ADB executable path.

        Searches for ADB in the following order (highest priority first):
        1. Local platform-tools folder (included in repository)
        2. System PATH
        3. Common BlueStacks installation directories
        4. Android SDK platform-tools directory

        Returns
        -------
        Optional[str]
            Path to adb.exe if found, None otherwise.
        """
        # Priority 1: Check local platform-tools folder (included in repo)
        # Get the project root by finding the directory containing platform-tools
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # Go up from memu_controller/ to project root
        local_adb_path = project_root / "platform-tools" / "adb.exe"
        if local_adb_path.exists():
            return str(local_adb_path)

        # Priority 2: Check if adb is in PATH
        adb_in_path = shutil.which("adb")
        if adb_in_path:
            return adb_in_path

        # Priority 3: Common BlueStacks installation paths
        bluestacks_paths = [
            Path("C:/Program Files/BlueStacks_nxt/adb.exe"),
            Path("C:/Program Files (x86)/BlueStacks/adb.exe"),
            Path(os.path.expanduser("~/AppData/Local/BlueStacks/adb.exe")),
            Path("C:/ProgramData/BlueStacks_nxt/adb.exe"),
        ]

        for bluestacks_path in bluestacks_paths:
            if bluestacks_path.exists():
                return str(bluestacks_path)

        # Priority 4: Check Android SDK platform-tools (common locations)
        sdk_paths = [
            Path(os.path.expanduser("~/AppData/Local/Android/Sdk/platform-tools/adb.exe")),
            Path("C:/Users/Public/Android/platform-tools/adb.exe"),
            Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools" / "adb.exe",
            Path(os.environ.get("ANDROID_SDK_ROOT", "")) / "platform-tools" / "adb.exe",
        ]

        for sdk_path in sdk_paths:
            if sdk_path.exists():
                return str(sdk_path)

        return None

