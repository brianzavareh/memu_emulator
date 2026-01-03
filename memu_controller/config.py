"""
Configuration management for MEmu controller.

This module handles configuration settings, including paths to MEmu executables
and default VM settings.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MemuConfig:
    """
    Configuration class for MEmu controller settings.

    Attributes
    ----------
    memuc_path : Optional[str]
        Path to memuc.exe. If None, pymemuc will attempt to auto-detect.
    default_vm_name : str
        Default name for created VMs.
    auto_start : bool
        Whether to automatically start VMs after creation.
    timeout : int
        Default timeout for operations in seconds.
    adb_port_base : int
        Base port for ADB connections (ports will be calculated as base + vm_index).
    """

    memuc_path: Optional[str] = None
    adb_path: Optional[str] = None
    default_vm_name: str = "Python_Controlled_VM"
    auto_start: bool = True
    timeout: int = 60
    adb_port_base: int = 21503

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemuConfig":
        """
        Create a MemuConfig instance from a dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration values.

        Returns
        -------
        MemuConfig
            Configured MemuConfig instance.
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
            "memuc_path": self.memuc_path,
            "adb_path": self.adb_path,
            "default_vm_name": self.default_vm_name,
            "auto_start": self.auto_start,
            "timeout": self.timeout,
            "adb_port_base": self.adb_port_base,
        }

    def get_adb_port(self, vm_index: int) -> int:
        """
        Calculate ADB port for a given VM index.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        int
            ADB port number for the VM.
        """
        return self.adb_port_base + vm_index

    @staticmethod
    def find_adb_path() -> Optional[str]:
        """
        Auto-detect ADB executable path.

        Searches for ADB in the following locations:
        1. System PATH
        2. Common MEmu installation directories
        3. Android SDK platform-tools directory

        Returns
        -------
        Optional[str]
            Path to adb.exe if found, None otherwise.
        """
        # Check if adb is in PATH
        adb_in_path = shutil.which("adb")
        if adb_in_path:
            return adb_in_path

        # Common MEmu installation paths
        memu_paths = [
            Path("C:/Program Files/Microvirt/MEmu/adb.exe"),
            Path("C:/Program Files (x86)/Microvirt/MEmu/adb.exe"),
            Path(os.path.expanduser("~/Microvirt/MEmu/adb.exe")),
        ]

        for memu_path in memu_paths:
            if memu_path.exists():
                return str(memu_path)

        # Check Android SDK platform-tools (common locations)
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

