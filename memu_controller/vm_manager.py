"""
BlueStacks instance management module.

This module handles BlueStacks instance detection and status checking via ADB.
Note: BlueStacks instances must be created and managed manually through the BlueStacks GUI.
"""

from typing import List, Optional, Dict, Any
import subprocess
import shutil


class VMManager:
    """
    Manager class for BlueStacks instance operations.

    This class provides a high-level interface for detecting and managing
    BlueStacks instances via ADB. Note that BlueStacks instances must be
    created and started manually through the BlueStacks application.

    Attributes
    ----------
    adb_path : Optional[str]
        Path to adb.exe. If None, attempts to auto-detect.
    """

    def __init__(self, adb_path: Optional[str] = None):
        """
        Initialize the VM Manager.

        Parameters
        ----------
        adb_path : Optional[str], optional
            Path to adb.exe. If None, attempts to auto-detect.
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

    def list_vms(self) -> List[Dict[str, Any]]:
        """
        List all available BlueStacks instances detected via ADB.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing instance information.
            Each dictionary contains 'index' and 'name' keys.
        """
        instances = []
        try:
            # Get connected ADB devices
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for idx, line in enumerate(lines):
                    if line.strip() and "device" in line:
                        device_id = line.split()[0]
                        # Check if it's a localhost connection (BlueStacks)
                        if "127.0.0.1" in device_id or "localhost" in device_id:
                            port = device_id.split(":")[-1] if ":" in device_id else "5555"
                            instances.append({
                                "index": idx,
                                "name": f"BlueStacks_{idx}",
                                "port": port,
                                "device_id": device_id
                            })
        except Exception as e:
            print(f"Error listing BlueStacks instances: {e}")
        
        return instances

    def get_vm_info(self, vm_index: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing instance information, or None if not found.
        """
        vms = self.list_vms()
        for vm in vms:
            if vm.get("index") == vm_index:
                return vm
        return None

    def create_vm(self, vm_name: Optional[str] = None) -> Optional[int]:
        """
        Create a new BlueStacks instance.

        Note: BlueStacks instances must be created manually through the
        BlueStacks application. This method is provided for API compatibility
        but will always return None.

        Parameters
        ----------
        vm_name : Optional[str], optional
            Name for the new instance (not used, for compatibility only).

        Returns
        -------
        Optional[int]
            Always returns None. Create instances manually in BlueStacks.
        """
        print("Note: BlueStacks instances must be created manually through the BlueStacks application.")
        return None

    def start_vm(self, vm_index: int) -> bool:
        """
        Start a BlueStacks instance.

        Note: BlueStacks instances must be started manually through the
        BlueStacks application. This method checks if the instance is running
        via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            True if instance is running (detected via ADB), False otherwise.
        """
        return self.is_vm_running(vm_index)

    def stop_vm(self, vm_index: int) -> bool:
        """
        Stop a running BlueStacks instance.

        Note: BlueStacks instances must be stopped manually through the
        BlueStacks application. This method is provided for API compatibility
        but does not actually stop the instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            Always returns True (for compatibility).
        """
        print("Note: BlueStacks instances must be stopped manually through the BlueStacks application.")
        return True

    def delete_vm(self, vm_index: int) -> bool:
        """
        Delete a BlueStacks instance.

        Note: BlueStacks instances must be deleted manually through the
        BlueStacks application. This method is provided for API compatibility
        but does not actually delete the instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            Always returns True (for compatibility).
        """
        print("Note: BlueStacks instances must be deleted manually through the BlueStacks application.")
        return True

    def is_vm_running(self, vm_index: int, adb_manager=None, adb_port_base: int = 5555) -> bool:
        """
        Check if a BlueStacks instance is currently running via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance to check.
        adb_manager : Optional[object], optional
            ADBManager instance to use for checking connectivity.
            If provided, will check ADB connectivity.
        adb_port_base : int, optional
            Base ADB port number. Default is 5555 for BlueStacks.

        Returns
        -------
        bool
            True if instance is running and accessible via ADB, False otherwise.
        """
        if adb_manager is not None:
            try:
                devices = adb_manager.get_connected_devices()
                port = adb_port_base if vm_index == 0 else adb_port_base + vm_index
                device_address = f"127.0.0.1:{port}"
                return device_address in devices
            except Exception:
                pass
        
        # Fallback: check directly via ADB
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                port = adb_port_base if vm_index == 0 else adb_port_base + vm_index
                device_address = f"127.0.0.1:{port}"
                return device_address in result.stdout
        except Exception:
            pass
        return False

    def wait_for_vm_ready(self, vm_index: int, timeout: int = 60, adb_port_base: int = 5555) -> bool:
        """
        Wait for a BlueStacks instance to be ready (booted and responsive via ADB).

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        timeout : int, optional
            Maximum time to wait in seconds. Default is 60.
        adb_port_base : int, optional
            Base ADB port number. Default is 5555.

        Returns
        -------
        bool
            True if instance is ready within timeout, False otherwise.
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_vm_running(vm_index, adb_port_base=adb_port_base):
                # Additional check: try to get instance info to ensure it's responsive
                if self.get_vm_info(vm_index) is not None:
                    # Try to execute a simple ADB command to verify responsiveness
                    port = adb_port_base if vm_index == 0 else adb_port_base + vm_index
                    try:
                        result = subprocess.run(
                            [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "getprop", "sys.boot_completed"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and "1" in result.stdout:
                            return True
                    except Exception:
                        pass
            time.sleep(2)
        
        return False

