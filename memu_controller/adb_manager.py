"""
ADB (Android Debug Bridge) management module.

This module handles ADB connections and operations for BlueStacks instances.
"""

from typing import Optional, List, Tuple, Dict
import subprocess
import time
import os
import shutil
from pathlib import Path


class ADBManager:
    """
    Manager class for ADB operations with BlueStacks instances.

    This class provides methods to connect to BlueStacks instances via ADB and execute
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
                from memu_controller.config import BlueStacksConfig
                detected_path = BlueStacksConfig.find_adb_path()
                if detected_path:
                    self.adb_path = detected_path
                else:
                    # Fallback to "adb" (will fail with clear error)
                    self.adb_path = "adb"

    def get_adb_port(self, vm_index: int, base_port: int = 5555) -> int:
        """
        Calculate ADB port for a given BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        base_port : int, optional
            Base port number. Default is 5555 (BlueStacks default).

        Returns
        -------
        int
            ADB port number for the instance.
        """
        # BlueStacks typically uses 5555 for the first instance
        if vm_index == 0:
            return base_port
        return base_port + vm_index

    def connect(self, vm_index: int, base_port: int = 5555) -> bool:
        """
        Connect to a BlueStacks instance via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        base_port : int, optional
            Base port number. Default is 5555.

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

    def disconnect(self, vm_index: int, base_port: int = 5555) -> bool:
        """
        Disconnect from a BlueStacks instance via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        base_port : int, optional
            Base port number. Default is 5555.

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

    def execute_command(self, vm_index: int, command: str, base_port: int = 5555) -> Optional[str]:
        """
        Execute an ADB shell command on a BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        command : str
            ADB shell command to execute.
        base_port : int, optional
            Base port number. Default is 5555.

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

    def install_apk(self, vm_index: int, apk_path: str, base_port: int = 5555) -> bool:
        """
        Install an APK on a BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        apk_path : str
            Path to the APK file to install.
        base_port : int, optional
            Base port number. Default is 5555.

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

    def wait_for_device(self, vm_index: int, timeout: int = 60, base_port: int = 5555) -> bool:
        """
        Wait for a BlueStacks instance to be available via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        timeout : int, optional
            Maximum time to wait in seconds. Default is 60.
        base_port : int, optional
            Base port number. Default is 5555.

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

    def take_screenshot(self, vm_index: int, save_path: Optional[str] = None, base_port: int = 5555) -> Optional[str]:
        """
        Take a screenshot of the BlueStacks instance screen.
        
        Uses WAKEUP refresh method to ensure the display buffer is refreshed before capturing.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        save_path : Optional[str], optional
            Path to save the screenshot. If None, saves to /sdcard/screenshot.png on device.
        base_port : int, optional
            Base port number. Default is 5555.

        Returns
        -------
        Optional[str]
            Path to the saved screenshot file, or None if failed.
        """
        port = self.get_adb_port(vm_index, base_port)
        
        # Use device path if no save path provided
        device_path = save_path or "/sdcard/screenshot.png"
        
        try:
            # Method 2 approach: Use WAKEUP to refresh display buffer (works with BlueStacks)
            refresh_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "input", "keyevent", "KEYCODE_WAKEUP"]
            subprocess.run(refresh_cmd, capture_output=True, timeout=5)
            
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


    def take_screenshot_bytes(self, vm_index: int, base_port: int = 5555, refresh_display: bool = True) -> Optional[bytes]:
        """
        Take a screenshot and return as bytes.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        base_port : int, optional
            Base port number. Default is 5555.
        refresh_display : bool, optional
            Whether to refresh the display before taking screenshot. 
            If False, takes screenshot without any interactions (no wakeup, no swipe).
            Default is True for backward compatibility.

        Returns
        -------
        Optional[bytes]
            Screenshot image as bytes, or None if failed.
        """
        port = self.get_adb_port(vm_index, base_port)
        
        # Only refresh display if requested (for non-interactive screenshots, skip this)
        if refresh_display:
            # Wake up screen and force refresh before taking screenshot
            # This is especially important for multi-monitor setups where BlueStacks might not update framebuffer
            # Note: We avoid sending BACK/HOME keys as they would close the current app
            try:
                # Force screen update by toggling display
                refresh_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "input", "keyevent", "KEYCODE_WAKEUP"]
                subprocess.run(refresh_cmd, capture_output=True, timeout=5)
                # Small swipe to trigger screen refresh (minimal movement to avoid app interaction)
                swipe_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "input", "swipe", "100", "100", "101", "101", "50"]
                subprocess.run(swipe_cmd, capture_output=True, timeout=5)
            except Exception as e:
                pass  # Ignore refresh errors, continue with screenshot
        # Try multiple screenshot methods to work around BlueStacks rendering issues
        # Based on research: BlueStacks may have issues with carriage returns and display IDs
        
        # First, check for available displays
        display_id = None
        try:
            display_check_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "dumpsys", "SurfaceFlinger", "--display-id"]
            display_result = subprocess.run(display_check_cmd, capture_output=True, timeout=5, text=True)
            if display_result.returncode == 0 and display_result.stdout:
                # Try to extract display ID (usually 0 for main display)
                import re
                display_ids = re.findall(r'Display\s+(\d+)', display_result.stdout)
                if display_ids:
                    display_id = display_ids[0]
        except Exception:
            pass  # Continue without display ID
        
        # Method 1: Use shell screencap to device, then pull (avoids text conversion issues)
        # This method writes directly to a file on the device and pulls it, avoiding
        # any carriage return corruption that can happen with exec-out on Windows
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), f"screenshot_{vm_index}_{int(time.time())}.png")
        device_path = "/sdcard/screenshot.png"
        
        result = None
        try:
            # Step 1: Take screenshot to device storage
            # ADB screencap syntax: screencap [-p] [-d display-id] <filename>
            screencap_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "screencap", "-p"]
            if display_id:
                screencap_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "screencap", "-p", "-d", display_id]
            screencap_cmd.append(device_path)
            
            screencap_result = subprocess.run(screencap_cmd, capture_output=True, timeout=10)
            
            if screencap_result.returncode == 0:
                # Step 2: Pull the file from device
                pull_cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "pull", device_path, tmp_path]
                pull_result = subprocess.run(pull_cmd, capture_output=True, timeout=10)
                
                if pull_result.returncode == 0 and os.path.exists(tmp_path):
                    # Step 3: Read the file
                    with open(tmp_path, "rb") as f:
                        screenshot_data = f.read()
                    # Clean up
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    # Clean up device file
                    try:
                        subprocess.run([self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "rm", device_path], 
                                     capture_output=True, timeout=5)
                    except:
                        pass
                    
                    result = type('obj', (object,), {'returncode': 0, 'stdout': screenshot_data})()
                else:
                    # Fallback to exec-out method
                    cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "exec-out", "screencap", "-p"]
                    if display_id:
                        cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "exec-out", "screencap", "-d", display_id, "-p"]
                    result = subprocess.run(cmd, capture_output=True, timeout=10)
                    # DO NOT process carriage returns - PNG is binary, modifying it corrupts the file
            else:
                # Fallback to exec-out method
                cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "exec-out", "screencap", "-p"]
                if display_id:
                    cmd = [self.adb_path, "-s", f"127.0.0.1:{port}", "exec-out", "screencap", "-d", display_id, "-p"]
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                # DO NOT process carriage returns - PNG is binary, modifying it corrupts the file
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None
        
        # Check result and return
        if result and result.returncode == 0:
            return result.stdout
        else:
            if result:
                print(f"Screenshot failed: {result.stderr}")
            return None

    def get_screen_size(self, vm_index: int, base_port: int = 5555) -> Optional[Tuple[int, int]]:
        """
        Get the screen size (width, height) of the BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        base_port : int, optional
            Base port number. Default is 5555.

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

