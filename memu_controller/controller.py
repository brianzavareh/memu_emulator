"""
Main controller module for MEmu emulator operations.

This module provides a high-level, unified interface for managing MEmu
emulators, combining VM management and ADB operations.
"""

from typing import Optional, Dict, Any, List, Tuple
import time
from memu_controller.config import MemuConfig
from memu_controller.vm_manager import VMManager
from memu_controller.adb_manager import ADBManager
from memu_controller.input_manager import InputManager
from memu_controller.image_utils import ImageProcessor
from PIL import Image


class MemuController:
    """
    Main controller class for MEmu emulator operations.

    This class provides a unified interface for managing MEmu VMs and
    performing ADB operations. It combines VM management and ADB functionality
    into a single, easy-to-use interface.

    Attributes
    ----------
    config : MemuConfig
        Configuration settings for the controller.
    vm_manager : VMManager
        Manager for VM operations.
    adb_manager : ADBManager
        Manager for ADB operations.
    active_vm_index : Optional[int]
        Currently active VM index.
    """

    def __init__(self, config: Optional[MemuConfig] = None):
        """
        Initialize the MEmu Controller.

        Parameters
        ----------
        config : Optional[MemuConfig], optional
            Configuration object. If None, default configuration is used.
        """
        self.config = config or MemuConfig()
        self.vm_manager = VMManager(memuc_path=self.config.memuc_path)
        # Use config adb_path or auto-detect
        adb_path = self.config.adb_path or MemuConfig.find_adb_path()
        self.adb_manager = ADBManager(adb_path=adb_path)
        self.input_manager = InputManager(adb_path=adb_path)
        self.image_processor = ImageProcessor()
        self.active_vm_index: Optional[int] = None

    def list_vms(self) -> List[Dict[str, Any]]:
        """
        List all available virtual machines.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing VM information.
        """
        return self.vm_manager.list_vms()

    def create_and_start_vm(self, vm_name: Optional[str] = None) -> Optional[int]:
        """
        Create a new VM and optionally start it.

        Parameters
        ----------
        vm_name : Optional[str], optional
            Name for the new VM. If None, uses default from config.

        Returns
        -------
        Optional[int]
            Index of the created VM, or None if creation failed.
        """
        name = vm_name or self.config.default_vm_name
        vm_index = self.vm_manager.create_vm(vm_name=name)
        
        if vm_index is not None and self.config.auto_start:
            if self.vm_manager.start_vm(vm_index):
                self.active_vm_index = vm_index
                # Wait for VM to be ready
                if self.vm_manager.wait_for_vm_ready(vm_index, self.config.timeout):
                    return vm_index
                else:
                    print(f"Warning: VM {vm_index} started but may not be fully ready")
                    return vm_index
            else:
                print(f"Warning: VM {vm_index} created but failed to start")
                return vm_index
        
        return vm_index

    def start_vm(self, vm_index: int, connect_adb: bool = True) -> bool:
        """
        Start a VM and optionally connect via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to start.
        connect_adb : bool, optional
            Whether to automatically connect via ADB after starting. Default is True.

        Returns
        -------
        bool
            True if VM was started successfully, False otherwise.
        """
        if self.vm_manager.start_vm(vm_index):
            self.active_vm_index = vm_index
            
            if connect_adb:
                # Wait for VM to be ready before connecting ADB
                if self.vm_manager.wait_for_vm_ready(vm_index, self.config.timeout):
                    time.sleep(3)  # Additional wait for ADB service
                    return self.adb_manager.connect(
                        vm_index, 
                        self.config.adb_port_base
                    )
                else:
                    print(f"Warning: VM {vm_index} started but ADB connection may fail")
                    return self.adb_manager.connect(vm_index, self.config.adb_port_base)
            
            return True
        return False

    def stop_vm(self, vm_index: int, disconnect_adb: bool = True) -> bool:
        """
        Stop a VM and optionally disconnect ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to stop.
        disconnect_adb : bool, optional
            Whether to disconnect ADB before stopping. Default is True.

        Returns
        -------
        bool
            True if VM was stopped successfully, False otherwise.
        """
        if disconnect_adb:
            self.adb_manager.disconnect(vm_index, self.config.adb_port_base)
        
        if self.vm_manager.stop_vm(vm_index):
            if self.active_vm_index == vm_index:
                self.active_vm_index = None
            return True
        return False

    def delete_vm(self, vm_index: int, force_stop: bool = True) -> bool:
        """
        Delete a VM, optionally stopping it first if running.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to delete.
        force_stop : bool, optional
            Whether to stop the VM before deletion if it's running. Default is True.

        Returns
        -------
        bool
            True if VM was deleted successfully, False otherwise.
        """
        if force_stop and self.vm_manager.is_vm_running(vm_index):
            self.stop_vm(vm_index, disconnect_adb=True)
        
        if self.vm_manager.delete_vm(vm_index):
            if self.active_vm_index == vm_index:
                self.active_vm_index = None
            return True
        return False

    def connect_adb(self, vm_index: int) -> bool:
        """
        Connect to a VM via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        return self.adb_manager.connect(vm_index, self.config.adb_port_base)

    def disconnect_adb(self, vm_index: int) -> bool:
        """
        Disconnect from a VM via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        bool
            True if disconnection successful, False otherwise.
        """
        return self.adb_manager.disconnect(vm_index, self.config.adb_port_base)

    def execute_adb_command(self, vm_index: int, command: str) -> Optional[str]:
        """
        Execute an ADB shell command on a VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        command : str
            ADB shell command to execute.

        Returns
        -------
        Optional[str]
            Command output, or None if execution failed.
        """
        return self.adb_manager.execute_command(
            vm_index, 
            command, 
            self.config.adb_port_base
        )

    def install_apk(self, vm_index: int, apk_path: str) -> bool:
        """
        Install an APK on a VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        apk_path : str
            Path to the APK file to install.

        Returns
        -------
        bool
            True if installation successful, False otherwise.
        """
        return self.adb_manager.install_apk(
            vm_index, 
            apk_path, 
            self.config.adb_port_base
        )

    def get_vm_status(self, vm_index: int) -> Dict[str, Any]:
        """
        Get comprehensive status information about a VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing VM status information including:
            - 'index': VM index
            - 'running': Whether VM is running
            - 'adb_connected': Whether ADB is connected
            - 'info': VM information dictionary
        """
        vm_info = self.vm_manager.get_vm_info(vm_index)
        is_running = self.vm_manager.is_vm_running(vm_index)
        
        adb_connected = False
        if is_running:
            devices = self.adb_manager.get_connected_devices()
            port = self.config.get_adb_port(vm_index)
            adb_connected = f"127.0.0.1:{port}" in devices
        
        return {
            "index": vm_index,
            "running": is_running,
            "adb_connected": adb_connected,
            "info": vm_info,
        }

    def set_active_vm(self, vm_index: int) -> bool:
        """
        Set a VM as the active VM for operations.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        bool
            True if VM exists and was set as active, False otherwise.
        """
        if self.vm_manager.get_vm_info(vm_index) is not None:
            self.active_vm_index = vm_index
            return True
        return False

    def get_active_vm(self) -> Optional[int]:
        """
        Get the currently active VM index.

        Returns
        -------
        Optional[int]
            Active VM index, or None if no VM is active.
        """
        return self.active_vm_index

    def take_screenshot(self, vm_index: int, save_path: Optional[str] = None) -> Optional[str]:
        """
        Take a screenshot of the VM screen.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        save_path : Optional[str], optional
            Path to save the screenshot locally. If None, saves to device only.

        Returns
        -------
        Optional[str]
            Path to the saved screenshot file, or None if failed.
        """
        return self.adb_manager.take_screenshot(
            vm_index,
            save_path,
            self.config.adb_port_base
        )

    def take_screenshot_image(self, vm_index: int) -> Optional[Image.Image]:
        """
        Take a screenshot and return as PIL Image.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        Optional[Image.Image]
            Screenshot as PIL Image, or None if failed.
        """
        screenshot_bytes = self.adb_manager.take_screenshot_bytes(
            vm_index,
            self.config.adb_port_base
        )
        if screenshot_bytes:
            return self.image_processor.load_from_bytes(screenshot_bytes)
        return None

    def get_screen_size(self, vm_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the screen size (width, height) of the VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        Optional[Tuple[int, int]]
            Tuple of (width, height), or None if failed.
        """
        return self.adb_manager.get_screen_size(vm_index, self.config.adb_port_base)

    def tap(self, vm_index: int, x: int, y: int) -> bool:
        """
        Perform a tap (touch) at the specified coordinates.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        bool
            True if tap was successful, False otherwise.
        """
        return self.input_manager.tap(vm_index, x, y, self.config.adb_port_base)

    def swipe(
        self,
        vm_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300
    ) -> bool:
        """
        Perform a swipe gesture from (x1, y1) to (x2, y2).

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        x1 : int
            Start X coordinate.
        y1 : int
            Start Y coordinate.
        x2 : int
            End X coordinate.
        y2 : int
            End Y coordinate.
        duration_ms : int, optional
            Duration of the swipe in milliseconds. Default is 300.

        Returns
        -------
        bool
            True if swipe was successful, False otherwise.
        """
        return self.input_manager.swipe(
            vm_index, x1, y1, x2, y2, duration_ms, self.config.adb_port_base
        )

    def long_press(self, vm_index: int, x: int, y: int, duration_ms: int = 500) -> bool:
        """
        Perform a long press at the specified coordinates.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        x : int
            X coordinate.
        y : int
            Y coordinate.
        duration_ms : int, optional
            Duration of the press in milliseconds. Default is 500.

        Returns
        -------
        bool
            True if long press was successful, False otherwise.
        """
        return self.input_manager.long_press(
            vm_index, x, y, duration_ms, self.config.adb_port_base
        )

    def drag(
        self,
        vm_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        steps: int = 10
    ) -> bool:
        """
        Perform a drag gesture with multiple steps for smooth movement.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        x1 : int
            Start X coordinate.
        y1 : int
            Start Y coordinate.
        x2 : int
            End X coordinate.
        y2 : int
            End Y coordinate.
        steps : int, optional
            Number of steps for smooth drag. Default is 10.

        Returns
        -------
        bool
            True if drag was successful, False otherwise.
        """
        return self.input_manager.drag(
            vm_index, x1, y1, x2, y2, steps, self.config.adb_port_base
        )

    def back(self, vm_index: int) -> bool:
        """
        Press the back button.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        bool
            True if back button press was successful, False otherwise.
        """
        return self.input_manager.back(vm_index, self.config.adb_port_base)

    def home(self, vm_index: int) -> bool:
        """
        Press the home button.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        bool
            True if home button press was successful, False otherwise.
        """
        return self.input_manager.home(vm_index, self.config.adb_port_base)

    def find_template_in_screenshot(
        self,
        vm_index: int,
        template_path: str,
        threshold: float = 0.8
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Take a screenshot and find a template image within it.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        template_path : str
            Path to the template image file.
        threshold : float, optional
            Matching threshold (0.0 to 1.0). Default is 0.8.

        Returns
        -------
        Optional[Tuple[int, int, int, int]]
            Tuple of (x, y, width, height) of found template, or None if not found.
        """
        screenshot = self.take_screenshot_image(vm_index)
        if screenshot:
            return self.image_processor.find_template(screenshot, template_path, threshold)
        return None

    def tap_template(self, vm_index: int, template_path: str, threshold: float = 0.8) -> bool:
        """
        Find a template in the screenshot and tap on its center.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        template_path : str
            Path to the template image file.
        threshold : float, optional
            Matching threshold (0.0 to 1.0). Default is 0.8.

        Returns
        -------
        bool
            True if template was found and tapped, False otherwise.
        """
        bbox = self.find_template_in_screenshot(vm_index, template_path, threshold)
        if bbox:
            center = self.image_processor.get_center(bbox)
            return self.tap(vm_index, center[0], center[1])
        return False

