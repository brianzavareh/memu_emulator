"""
Main controller module for BlueStacks emulator operations.

This module provides a high-level, unified interface for managing BlueStacks
instances, combining instance detection and ADB operations.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import time
from memu_controller.config import BlueStacksConfig
from memu_controller.vm_manager import VMManager
from memu_controller.adb_manager import ADBManager
from memu_controller.input_manager import InputManager
from memu_controller.image_utils import ImageProcessor
from memu_controller.coordinate_utils import CoordinateConverter
from PIL import Image


class BlueStacksController:
    """
    Main controller class for BlueStacks emulator operations.

    This class provides a unified interface for managing BlueStacks instances and
    performing ADB operations. It combines instance detection and ADB functionality
    into a single, easy-to-use interface.

    Attributes
    ----------
    config : BlueStacksConfig
        Configuration settings for the controller.
    vm_manager : VMManager
        Manager for instance operations.
    adb_manager : ADBManager
        Manager for ADB operations.
    active_vm_index : Optional[int]
        Currently active instance index.
    """

    def __init__(self, config: Optional[BlueStacksConfig] = None):
        """
        Initialize the BlueStacks Controller.

        Parameters
        ----------
        config : Optional[BlueStacksConfig], optional
            Configuration object. If None, default configuration is used.
        """
        self.config = config or BlueStacksConfig()
        # Use config adb_path or auto-detect
        adb_path = self.config.adb_path or BlueStacksConfig.find_adb_path()
        self.vm_manager = VMManager(adb_path=adb_path)
        self.adb_manager = ADBManager(adb_path=adb_path)
        self.input_manager = InputManager(adb_path=adb_path)
        self.image_processor = ImageProcessor()
        self.active_vm_index: Optional[int] = None

    def list_vms(self) -> List[Dict[str, Any]]:
        """
        List all available BlueStacks instances detected via ADB.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing instance information.
        """
        return self.vm_manager.list_vms()

    def create_and_start_vm(self, vm_name: Optional[str] = None) -> Optional[int]:
        """
        Create a new BlueStacks instance and optionally start it.

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

    def start_vm(self, vm_index: int, connect_adb: bool = True) -> bool:
        """
        Check if a BlueStacks instance is running and optionally connect via ADB.

        Note: BlueStacks instances must be started manually through the
        BlueStacks application. This method checks if the instance is running
        and connects via ADB if requested.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        connect_adb : bool, optional
            Whether to automatically connect via ADB. Default is True.

        Returns
        -------
        bool
            True if instance is running and ADB connection successful (if requested), False otherwise.
        """
        if self.vm_manager.start_vm(vm_index):
            self.active_vm_index = vm_index
            
            if connect_adb:
                # Wait for instance to be ready before connecting ADB
                if self.vm_manager.wait_for_vm_ready(vm_index, self.config.timeout, self.config.adb_port_base):
                    time.sleep(3)  # Additional wait for ADB service
                    return self.adb_manager.connect(
                        vm_index, 
                        self.config.adb_port_base
                    )
                else:
                    print(f"Warning: BlueStacks instance {vm_index} may not be fully ready")
                    return self.adb_manager.connect(vm_index, self.config.adb_port_base)
            
            return True
        return False

    def stop_vm(self, vm_index: int, disconnect_adb: bool = True) -> bool:
        """
        Disconnect ADB from a BlueStacks instance.

        Note: BlueStacks instances must be stopped manually through the
        BlueStacks application. This method only disconnects ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        disconnect_adb : bool, optional
            Whether to disconnect ADB. Default is True.

        Returns
        -------
        bool
            True if ADB was disconnected successfully, False otherwise.
        """
        if disconnect_adb:
            self.adb_manager.disconnect(vm_index, self.config.adb_port_base)
        
        if self.active_vm_index == vm_index:
            self.active_vm_index = None
        return True

    def delete_vm(self, vm_index: int, force_stop: bool = True) -> bool:
        """
        Delete a BlueStacks instance.

        Note: BlueStacks instances must be deleted manually through the
        BlueStacks application. This method is provided for API compatibility
        but does not actually delete the instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        force_stop : bool, optional
            Whether to disconnect ADB before deletion. Default is True.

        Returns
        -------
        bool
            Always returns True (for compatibility).
        """
        if force_stop:
            self.stop_vm(vm_index, disconnect_adb=True)
        
        if self.active_vm_index == vm_index:
            self.active_vm_index = None
        return True

    def connect_adb(self, vm_index: int) -> bool:
        """
        Connect to a BlueStacks instance via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        return self.adb_manager.connect(vm_index, self.config.adb_port_base)

    def disconnect_adb(self, vm_index: int) -> bool:
        """
        Disconnect from a BlueStacks instance via ADB.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            True if disconnection successful, False otherwise.
        """
        return self.adb_manager.disconnect(vm_index, self.config.adb_port_base)

    def execute_adb_command(self, vm_index: int, command: str) -> Optional[str]:
        """
        Execute an ADB shell command on a BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
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
        Install an APK on a BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
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
        Get comprehensive status information about a BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing instance status information including:
            - 'index': Instance index
            - 'running': Whether instance is running (detected via ADB)
            - 'adb_connected': Whether ADB is connected
            - 'info': Instance information dictionary
        """
        vm_info = self.vm_manager.get_vm_info(vm_index)
        # Check if instance is running via ADB
        is_running = self.vm_manager.is_vm_running(vm_index, adb_manager=self.adb_manager, adb_port_base=self.config.adb_port_base)
        
        # Always check ADB connection status
        devices = self.adb_manager.get_connected_devices()
        port = self.config.get_adb_port(vm_index)
        adb_connected = f"127.0.0.1:{port}" in devices
        
        # If device is not detected, attempt to connect to BlueStacks
        # This handles the case where BlueStacks is running but ADB is not connected
        if not is_running:
            # Attempt to connect (this is idempotent - won't fail if already connected)
            connect_result = self.adb_manager.connect(vm_index, self.config.adb_port_base)
            
            # Re-check device status after connection attempt
            if connect_result:
                import time
                time.sleep(1)  # Brief wait for connection to register
                devices_after = self.adb_manager.get_connected_devices()
                adb_connected = f"127.0.0.1:{port}" in devices_after
                is_running = adb_connected or self.vm_manager.is_vm_running(vm_index, adb_manager=self.adb_manager, adb_port_base=self.config.adb_port_base)
        return {
            "index": vm_index,
            "running": is_running,
            "adb_connected": adb_connected,
            "info": vm_info,
        }

    def set_active_vm(self, vm_index: int) -> bool:
        """
        Set a BlueStacks instance as the active instance for operations.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        bool
            True if instance exists and was set as active, False otherwise.
        """
        if self.vm_manager.get_vm_info(vm_index) is not None:
            self.active_vm_index = vm_index
            return True
        return False

    def get_active_vm(self) -> Optional[int]:
        """
        Get the currently active BlueStacks instance index.

        Returns
        -------
        Optional[int]
            Active instance index, or None if no instance is active.
        """
        return self.active_vm_index

    def take_screenshot(self, vm_index: int, save_path: Optional[str] = None) -> Optional[str]:
        """
        Take a screenshot of the BlueStacks instance screen.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
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

    def take_screenshot_image(self, vm_index: int, refresh_display: bool = True) -> Optional[Image.Image]:
        """
        Take a screenshot and return as PIL Image.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        refresh_display : bool, optional
            Whether to refresh the display before taking screenshot.
            If False, takes screenshot without any interactions (no wakeup, no swipe).
            Default is True for backward compatibility.

        Returns
        -------
        Optional[Image.Image]
            Screenshot as PIL Image, or None if failed.
        """
        screenshot_bytes = self.adb_manager.take_screenshot_bytes(
            vm_index,
            self.config.adb_port_base,
            refresh_display=refresh_display
        )
        if screenshot_bytes:
            return self.image_processor.load_from_bytes(screenshot_bytes)
        return None


    def get_screen_size(self, vm_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the screen size (width, height) of the BlueStacks instance.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

        Returns
        -------
        Optional[Tuple[int, int]]
            Tuple of (width, height), or None if failed.
        """
        return self.adb_manager.get_screen_size(vm_index, self.config.adb_port_base)

    def tap(
        self,
        vm_index: int,
        x: Union[int, float, Tuple[Union[int, float], Union[int, float]]],
        y: Optional[Union[int, float]] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> bool:
        """
        Perform a tap (touch) at the specified coordinates.

        Supports both exact (int) and relational (float 0.0-1.0) coordinates.
        If relational coordinates are used, screen_width and screen_height must be provided,
        or use tap_on_screenshot() method which automatically uses screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x : Union[int, float, Tuple[Union[int, float], Union[int, float]]]
            X coordinate, or tuple of (x, y) coordinates.
            If float, represents percentage (0.0-1.0) of screen width.
            If int, represents exact pixel position.
        y : Optional[Union[int, float]], optional
            Y coordinate. If x is a tuple, this parameter is ignored.
            If float, represents percentage (0.0-1.0) of screen height.
            If int, represents exact pixel position.
        screen_width : Optional[int], optional
            Screen width in pixels. Required if using relational coordinates.
        screen_height : Optional[int], optional
            Screen height in pixels. Required if using relational coordinates.

        Returns
        -------
        bool
            True if tap was successful, False otherwise.

        Examples
        --------
        >>> # Exact coordinates
        >>> controller.tap(0, 100, 200)
        >>> # Relational coordinates (center of screen)
        >>> controller.tap(0, 0.5, 0.5, screen_width=1920, screen_height=1080)
        >>> # Using tuple
        >>> controller.tap(0, (0.5, 0.5), screen_width=1920, screen_height=1080)
        """
        return self.input_manager.tap(
            vm_index, x, y,
            screen_width=screen_width,
            screen_height=screen_height,
            base_port=self.config.adb_port_base
        )

    def swipe(
        self,
        vm_index: int,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        duration_ms: int = 300,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> bool:
        """
        Perform a swipe gesture from (x1, y1) to (x2, y2).

        Supports both exact (int) and relational (float 0.0-1.0) coordinates.
        If relational coordinates are used, screen_width and screen_height must be provided,
        or use swipe_on_screenshot() method which automatically uses screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x1 : Union[int, float]
            Start X coordinate. If float, represents percentage (0.0-1.0) of screen width.
        y1 : Union[int, float]
            Start Y coordinate. If float, represents percentage (0.0-1.0) of screen height.
        x2 : Union[int, float]
            End X coordinate. If float, represents percentage (0.0-1.0) of screen width.
        y2 : Union[int, float]
            End Y coordinate. If float, represents percentage (0.0-1.0) of screen height.
        duration_ms : int, optional
            Duration of the swipe in milliseconds. Default is 300.
        screen_width : Optional[int], optional
            Screen width in pixels. Required if using relational coordinates.
        screen_height : Optional[int], optional
            Screen height in pixels. Required if using relational coordinates.

        Returns
        -------
        bool
            True if swipe was successful, False otherwise.

        Examples
        --------
        >>> # Exact coordinates
        >>> controller.swipe(0, 100, 200, 300, 400)
        >>> # Relational coordinates (swipe from left to right)
        >>> controller.swipe(0, 0.1, 0.5, 0.9, 0.5, screen_width=1920, screen_height=1080)
        """
        return self.input_manager.swipe(
            vm_index, x1, y1, x2, y2, duration_ms,
            screen_width=screen_width,
            screen_height=screen_height,
            base_port=self.config.adb_port_base
        )

    def long_press(
        self,
        vm_index: int,
        x: Union[int, float],
        y: Union[int, float],
        duration_ms: int = 500,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> bool:
        """
        Perform a long press at the specified coordinates.

        Supports both exact (int) and relational (float 0.0-1.0) coordinates.
        If relational coordinates are used, screen_width and screen_height must be provided,
        or use long_press_on_screenshot() method which automatically uses screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x : Union[int, float]
            X coordinate. If float, represents percentage (0.0-1.0) of screen width.
        y : Union[int, float]
            Y coordinate. If float, represents percentage (0.0-1.0) of screen height.
        duration_ms : int, optional
            Duration of the press in milliseconds. Default is 500.
        screen_width : Optional[int], optional
            Screen width in pixels. Required if using relational coordinates.
        screen_height : Optional[int], optional
            Screen height in pixels. Required if using relational coordinates.

        Returns
        -------
        bool
            True if long press was successful, False otherwise.

        Examples
        --------
        >>> # Exact coordinates
        >>> controller.long_press(0, 100, 200)
        >>> # Relational coordinates (center of screen)
        >>> controller.long_press(0, 0.5, 0.5, screen_width=1920, screen_height=1080)
        """
        return self.input_manager.long_press(
            vm_index, x, y, duration_ms,
            screen_width=screen_width,
            screen_height=screen_height,
            base_port=self.config.adb_port_base
        )

    def drag(
        self,
        vm_index: int,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        steps: int = 10,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> bool:
        """
        Perform a drag gesture with multiple steps for smooth movement.

        Supports both exact (int) and relational (float 0.0-1.0) coordinates.
        If relational coordinates are used, screen_width and screen_height must be provided,
        or use drag_on_screenshot() method which automatically uses screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x1 : Union[int, float]
            Start X coordinate. If float, represents percentage (0.0-1.0) of screen width.
        y1 : Union[int, float]
            Start Y coordinate. If float, represents percentage (0.0-1.0) of screen height.
        x2 : Union[int, float]
            End X coordinate. If float, represents percentage (0.0-1.0) of screen width.
        y2 : Union[int, float]
            End Y coordinate. If float, represents percentage (0.0-1.0) of screen height.
        steps : int, optional
            Number of steps for smooth drag. Default is 10.
        screen_width : Optional[int], optional
            Screen width in pixels. Required if using relational coordinates.
        screen_height : Optional[int], optional
            Screen height in pixels. Required if using relational coordinates.

        Returns
        -------
        bool
            True if drag was successful, False otherwise.

        Examples
        --------
        >>> # Exact coordinates
        >>> controller.drag(0, 100, 200, 300, 400)
        >>> # Relational coordinates (drag from top-left to bottom-right)
        >>> controller.drag(0, 0.1, 0.1, 0.9, 0.9, screen_width=1920, screen_height=1080)
        """
        return self.input_manager.drag(
            vm_index, x1, y1, x2, y2, steps,
            screen_width=screen_width,
            screen_height=screen_height,
            base_port=self.config.adb_port_base
        )

    def back(self, vm_index: int) -> bool:
        """
        Press the back button.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.

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
            Index of the BlueStacks instance.

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
            Index of the BlueStacks instance.
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
            Index of the BlueStacks instance.
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

    def tap_on_screenshot(
        self,
        vm_index: int,
        x: Union[int, float, Tuple[Union[int, float], Union[int, float]]],
        y: Optional[Union[int, float]] = None,
        screenshot: Optional[Image.Image] = None
    ) -> bool:
        """
        Perform a tap using coordinates relative to a screenshot.

        Takes a screenshot (or uses provided one) and converts relational coordinates
        based on the screenshot dimensions. This is ideal for modular game automation
        where you analyze screenshots and decide actions based on their content.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x : Union[int, float, Tuple[Union[int, float], Union[int, float]]]
            X coordinate, or tuple of (x, y) coordinates.
            If float, represents percentage (0.0-1.0) of screenshot width.
            If int, represents exact pixel position in screenshot.
        y : Optional[Union[int, float]], optional
            Y coordinate. If x is a tuple, this parameter is ignored.
            If float, represents percentage (0.0-1.0) of screenshot height.
            If int, represents exact pixel position in screenshot.
        screenshot : Optional[Image.Image], optional
            Screenshot image to use for dimension reference.
            If None, a new screenshot will be taken.

        Returns
        -------
        bool
            True if tap was successful, False otherwise.

        Examples
        --------
        >>> # Tap center of screenshot
        >>> controller.tap_on_screenshot(0, 0.5, 0.5)
        >>> # Tap using tuple
        >>> controller.tap_on_screenshot(0, (0.5, 0.5))
        >>> # Use existing screenshot
        >>> screenshot = controller.take_screenshot_image(0)
        >>> controller.tap_on_screenshot(0, 0.3, 0.7, screenshot=screenshot)
        """
        if screenshot is None:
            screenshot = self.take_screenshot_image(vm_index)
            if screenshot is None:
                return False

        width, height = screenshot.size
        return self.tap(vm_index, x, y, screen_width=width, screen_height=height)

    def swipe_on_screenshot(
        self,
        vm_index: int,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        duration_ms: int = 300,
        screenshot: Optional[Image.Image] = None
    ) -> bool:
        """
        Perform a swipe using coordinates relative to a screenshot.

        Takes a screenshot (or uses provided one) and converts relational coordinates
        based on the screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x1 : Union[int, float]
            Start X coordinate. If float, represents percentage (0.0-1.0) of screenshot width.
        y1 : Union[int, float]
            Start Y coordinate. If float, represents percentage (0.0-1.0) of screenshot height.
        x2 : Union[int, float]
            End X coordinate. If float, represents percentage (0.0-1.0) of screenshot width.
        y2 : Union[int, float]
            End Y coordinate. If float, represents percentage (0.0-1.0) of screenshot height.
        duration_ms : int, optional
            Duration of the swipe in milliseconds. Default is 300.
        screenshot : Optional[Image.Image], optional
            Screenshot image to use for dimension reference.
            If None, a new screenshot will be taken.

        Returns
        -------
        bool
            True if swipe was successful, False otherwise.

        Examples
        --------
        >>> # Swipe from left to right (using relational coordinates)
        >>> controller.swipe_on_screenshot(0, 0.1, 0.5, 0.9, 0.5)
        """
        if screenshot is None:
            screenshot = self.take_screenshot_image(vm_index)
            if screenshot is None:
                return False

        width, height = screenshot.size
        return self.swipe(
            vm_index, x1, y1, x2, y2, duration_ms,
            screen_width=width, screen_height=height
        )

    def long_press_on_screenshot(
        self,
        vm_index: int,
        x: Union[int, float],
        y: Union[int, float],
        duration_ms: int = 500,
        screenshot: Optional[Image.Image] = None
    ) -> bool:
        """
        Perform a long press using coordinates relative to a screenshot.

        Takes a screenshot (or uses provided one) and converts relational coordinates
        based on the screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x : Union[int, float]
            X coordinate. If float, represents percentage (0.0-1.0) of screenshot width.
        y : Union[int, float]
            Y coordinate. If float, represents percentage (0.0-1.0) of screenshot height.
        duration_ms : int, optional
            Duration of the press in milliseconds. Default is 500.
        screenshot : Optional[Image.Image], optional
            Screenshot image to use for dimension reference.
            If None, a new screenshot will be taken.

        Returns
        -------
        bool
            True if long press was successful, False otherwise.

        Examples
        --------
        >>> # Long press at center of screenshot
        >>> controller.long_press_on_screenshot(0, 0.5, 0.5)
        """
        if screenshot is None:
            screenshot = self.take_screenshot_image(vm_index)
            if screenshot is None:
                return False

        width, height = screenshot.size
        return self.long_press(
            vm_index, x, y, duration_ms,
            screen_width=width, screen_height=height
        )

    def drag_on_screenshot(
        self,
        vm_index: int,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        steps: int = 10,
        screenshot: Optional[Image.Image] = None
    ) -> bool:
        """
        Perform a drag using coordinates relative to a screenshot.

        Takes a screenshot (or uses provided one) and converts relational coordinates
        based on the screenshot dimensions.

        Parameters
        ----------
        vm_index : int
            Index of the BlueStacks instance.
        x1 : Union[int, float]
            Start X coordinate. If float, represents percentage (0.0-1.0) of screenshot width.
        y1 : Union[int, float]
            Start Y coordinate. If float, represents percentage (0.0-1.0) of screenshot height.
        x2 : Union[int, float]
            End X coordinate. If float, represents percentage (0.0-1.0) of screenshot width.
        y2 : Union[int, float]
            End Y coordinate. If float, represents percentage (0.0-1.0) of screenshot height.
        steps : int, optional
            Number of steps for smooth drag. Default is 10.
        screenshot : Optional[Image.Image], optional
            Screenshot image to use for dimension reference.
            If None, a new screenshot will be taken.

        Returns
        -------
        bool
            True if drag was successful, False otherwise.

        Examples
        --------
        >>> # Drag from top-left to bottom-right (using relational coordinates)
        >>> controller.drag_on_screenshot(0, 0.1, 0.1, 0.9, 0.9)
        """
        if screenshot is None:
            screenshot = self.take_screenshot_image(vm_index)
            if screenshot is None:
                return False

        width, height = screenshot.size
        return self.drag(
            vm_index, x1, y1, x2, y2, steps,
            screen_width=width, screen_height=height
        )

