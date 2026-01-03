"""
ADB (Android Debug Bridge) management module.

This module handles ADB connections and operations for MEmu virtual machines.
"""

from typing import Optional, List, Tuple
import subprocess
import time
import os
import json
import shutil
from pathlib import Path


class ADBManager:
    """
    Manager class for ADB operations with MEmu VMs.

    This class provides methods to connect to VMs via ADB and execute
    ADB commands.

    Attributes
    ----------
    adb_path : Optional[str]
        Path to adb.exe. If None, assumes adb is in PATH.
    """

    def __init__(self, adb_path: Optional[str] = None):
        """
        Initialize the ADB Manager.

        Parameters
        ----------
        adb_path : Optional[str], optional
            Path to adb.exe. If None, attempts to auto-detect ADB.
        """
        if adb_path:
            self.adb_path = adb_path
        else:
            # Try to find ADB in PATH first
            found_path = shutil.which("adb")
            if found_path:
                self.adb_path = found_path
            else:
                # Auto-detect from common locations
                from memu_controller.config import MemuConfig
                detected_path = MemuConfig.find_adb_path()
                if detected_path:
                    self.adb_path = detected_path
                else:
                    # Fallback to "adb" (will fail with clear error)
                    self.adb_path = "adb"

    def get_adb_port(self, vm_index: int, base_port: int = 21503) -> int:
        """
        Calculate ADB port for a given VM index.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503 (MEmu default).

        Returns
        -------
        int
            ADB port number for the VM.
        """
        return base_port + vm_index

    def connect(self, vm_index: int, base_port: int = 21503) -> bool:
        """
        Connect to a VM via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        port = self.get_adb_port(vm_index, base_port)
        try:
            result = subprocess.run(
                [self.adb_path, "connect", f"127.0.0.1:{port}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "connected" in result.stdout.lower() or "already connected" in result.stdout.lower()
        except Exception as e:
            print(f"Error connecting to VM {vm_index} via ADB: {e}")
            return False

    def disconnect(self, vm_index: int, base_port: int = 21503) -> bool:
        """
        Disconnect from a VM via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if disconnection successful, False otherwise.
        """
        port = self.get_adb_port(vm_index, base_port)
        try:
            result = subprocess.run(
                [self.adb_path, "disconnect", f"127.0.0.1:{port}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error disconnecting from VM {vm_index} via ADB: {e}")
            return False

    def execute_command(self, vm_index: int, command: str, base_port: int = 21503) -> Optional[str]:
        """
        Execute an ADB shell command on a VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        command : str
            ADB shell command to execute.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        Optional[str]
            Command output, or None if execution failed.
        """
        port = self.get_adb_port(vm_index, base_port)
        try:
            result = subprocess.run(
                [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", command],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"ADB command failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error executing ADB command on VM {vm_index}: {e}")
            return None

    def install_apk(self, vm_index: int, apk_path: str, base_port: int = 21503) -> bool:
        """
        Install an APK on a VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        apk_path : str
            Path to the APK file to install.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if installation successful, False otherwise.
        """
        port = self.get_adb_port(vm_index, base_port)
        try:
            result = subprocess.run(
                [self.adb_path, "-s", f"127.0.0.1:{port}", "install", apk_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            return "Success" in result.stdout or result.returncode == 0
        except Exception as e:
            print(f"Error installing APK on VM {vm_index}: {e}")
            return False

    def get_connected_devices(self) -> List[str]:
        """
        Get list of connected ADB devices.

        Returns
        -------
        List[str]
            List of device identifiers.
        """
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            devices = []
            for line in result.stdout.split("\n")[1:]:  # Skip header
                if line.strip() and "device" in line:
                    devices.append(line.split()[0])
            return devices
        except Exception as e:
            print(f"Error getting connected devices: {e}")
            return []

    def wait_for_device(self, vm_index: int, timeout: int = 60, base_port: int = 21503) -> bool:
        """
        Wait for a device to be available via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        timeout : int, optional
            Maximum time to wait in seconds. Default is 60.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if device becomes available, False otherwise.
        """
        port = self.get_adb_port(vm_index, base_port)
        device_address = f"127.0.0.1:{port}"
        start_time = time.time()

        while time.time() - start_time < timeout:
            devices = self.get_connected_devices()
            if device_address in devices:
                return True
            time.sleep(2)

        return False

    def take_screenshot(self, vm_index: int, save_path: Optional[str] = None, base_port: int = 21503) -> Optional[str]:
        """
        Take a screenshot of the VM screen.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        save_path : Optional[str], optional
            Path to save the screenshot. If None, saves to /sdcard/screenshot.png on device.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        Optional[str]
            Path to the saved screenshot file, or None if failed.
        """
        port = self.get_adb_port(vm_index, base_port)
        
        # Use device path if no save path provided
        device_path = save_path or "/sdcard/screenshot.png"
        
        try:
            # Take screenshot on device
            result = subprocess.run(
                [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "screencap", "-p", device_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"Screenshot failed: {result.stderr}")
                return None
            
            # If save_path was provided, pull the file to local system
            if save_path and not save_path.startswith("/"):
                local_path = save_path
                # Pull file from device
                pull_result = subprocess.run(
                    [self.adb_path, "-s", f"127.0.0.1:{port}", "pull", device_path, local_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if pull_result.returncode == 0:
                    return local_path
                else:
                    print(f"Failed to pull screenshot: {pull_result.stderr}")
                    return None
            else:
                return device_path
        except Exception as e:
            print(f"Error taking screenshot on VM {vm_index}: {e}")
            return None

    def take_screenshot_bytes(self, vm_index: int, base_port: int = 21503) -> Optional[bytes]:
        """
        Take a screenshot and return as bytes.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        Optional[bytes]
            Screenshot image as bytes, or None if failed.
        """
        # #region agent log
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "adb_manager.py:308", "message": "take_screenshot_bytes entry", "data": {"vm_index": vm_index, "base_port": base_port, "adb_path": self.adb_path, "adb_path_exists": os.path.exists(self.adb_path) if self.adb_path != "adb" else None}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        port = self.get_adb_port(vm_index, base_port)
        # Wake up screen before taking screenshot
        # #region agent log
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            # Send key event to wake screen (KEYCODE_WAKEUP = 224)
            wake_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "input", "keyevent", "224"]
            wake_result = subprocess.run(wake_cmd, capture_output=True, timeout=5)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "G", "location": "adb_manager.py:335", "message": "Screen wake attempt", "data": {"returncode": wake_result.returncode, "stderr": wake_result.stderr.decode("utf-8", errors="replace") if wake_result.stderr else None}, "timestamp": int(time.time() * 1000)}) + "\n")
            time.sleep(0.5)  # Brief wait for screen to wake
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "G", "location": "adb_manager.py:340", "message": "Screen wake error", "data": {"error": str(e)}, "timestamp": int(time.time() * 1000)}) + "\n")
            pass  # Ignore wake errors, continue with screenshot
        # #endregion
        # #region agent log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "D", "location": "adb_manager.py:312", "message": "ADB port calculated", "data": {"port": port, "vm_index": vm_index, "base_port": base_port}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        # #region agent log
        adb_exists_check = shutil.which(self.adb_path) if self.adb_path == "adb" else os.path.exists(self.adb_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "adb_manager.py:316", "message": "ADB path check", "data": {"adb_path": self.adb_path, "adb_exists": adb_exists_check, "which_result": shutil.which(self.adb_path) if self.adb_path == "adb" else None}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        # #region agent log
        devices = self.get_connected_devices()
        device_address = f"127.0.0.1:{port}"
        is_connected = device_address in devices
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "adb_manager.py:322", "message": "ADB connection status", "data": {"devices": devices, "target_device": device_address, "is_connected": is_connected}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "exec-out", "screencap", "-p"]
        # #region agent log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "adb_manager.py:327", "message": "Before subprocess.run", "data": {"command": cmd}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10
            )
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "E", "location": "adb_manager.py:336", "message": "After subprocess.run", "data": {"returncode": result.returncode, "stdout_len": len(result.stdout) if result.stdout else 0, "stderr": result.stderr.decode("utf-8", errors="replace") if result.stderr else None}, "timestamp": int(time.time() * 1000)}) + "\n")
            # #endregion
            if result.returncode == 0:
                # #region agent log
                # Check if bytes look like PNG (starts with PNG signature)
                is_png = result.stdout[:8] == b'\x89PNG\r\n\x1a\n' if len(result.stdout) >= 8 else False
                first_bytes_hex = result.stdout[:16].hex() if len(result.stdout) >= 16 else result.stdout.hex()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "F", "location": "adb_manager.py:340", "message": "Screenshot success", "data": {"bytes_length": len(result.stdout), "is_png": is_png, "first_bytes_hex": first_bytes_hex}, "timestamp": int(time.time() * 1000)}) + "\n")
                # #endregion
                return result.stdout
            else:
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "E", "location": "adb_manager.py:344", "message": "Screenshot failed - non-zero returncode", "data": {"returncode": result.returncode, "stderr": result.stderr.decode("utf-8", errors="replace") if result.stderr else None}, "timestamp": int(time.time() * 1000)}) + "\n")
                # #endregion
                print(f"Screenshot failed: {result.stderr}")
                return None
        except FileNotFoundError as e:
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "adb_manager.py:350", "message": "FileNotFoundError exception", "data": {"error_type": type(e).__name__, "error_msg": str(e), "adb_path": self.adb_path}, "timestamp": int(time.time() * 1000)}) + "\n")
            # #endregion
            error_msg = (
                f"Error taking screenshot on VM {vm_index}: ADB executable not found.\n"
                f"  Attempted path: {self.adb_path}\n"
                f"  Please ensure ADB is installed and either:\n"
                f"  1. Add ADB to your system PATH, or\n"
                f"  2. Configure adb_path in MemuConfig, or\n"
                f"  3. Install ADB in a standard location (MEmu directory or Android SDK)"
            )
            print(error_msg)
            return None
        except Exception as e:
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "E", "location": "adb_manager.py:356", "message": "General exception", "data": {"error_type": type(e).__name__, "error_msg": str(e), "error_args": e.args if hasattr(e, "args") else None}, "timestamp": int(time.time() * 1000)}) + "\n")
            # #endregion
            print(f"Error taking screenshot on VM {vm_index}: {e}")
            return None

    def get_screen_size(self, vm_index: int, base_port: int = 21503) -> Optional[Tuple[int, int]]:
        """
        Get the screen size (width, height) of the VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        Optional[Tuple[int, int]]
            Tuple of (width, height), or None if failed.
        """
        result = self.execute_command(vm_index, "wm size", base_port)
        if result:
            try:
                # Parse output like "Physical size: 1920x1080"
                size_str = result.split(":")[-1].strip()
                width, height = map(int, size_str.split("x"))
                return (width, height)
            except (ValueError, IndexError):
                pass
        return None

