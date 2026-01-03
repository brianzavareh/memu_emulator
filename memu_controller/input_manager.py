"""
Input and gesture management module.

This module handles touch input, gestures, and other user interactions
with MEmu virtual machines via ADB.
"""

from typing import Optional, Tuple, List
import subprocess
import time
import shutil


class InputManager:
    """
    Manager class for input operations and gestures on MEmu VMs.

    This class provides methods for simulating touch events, swipes,
    long presses, and other gestures on the emulator screen.

    Attributes
    ----------
    adb_path : Optional[str]
        Path to adb.exe. If None, assumes adb is in PATH.
    """

    def __init__(self, adb_path: Optional[str] = None):
        """
        Initialize the Input Manager.

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
            Base port number. Default is 21503.

        Returns
        -------
        int
            ADB port number for the VM.
        """
        return base_port + vm_index

    def _execute_input_command(self, vm_index: int, command: str, base_port: int = 21503) -> bool:
        """
        Execute an input command on the VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        command : str
            Input command to execute.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if command executed successfully, False otherwise.
        """
        port = self.get_adb_port(vm_index, base_port)
        try:
            result = subprocess.run(
                [self.adb_path, "-s", f"127.0.0.1:{port}", "shell", "input", command],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error executing input command on VM {vm_index}: {e}")
            return False

    def tap(self, vm_index: int, x: int, y: int, base_port: int = 21503) -> bool:
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
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if tap was successful, False otherwise.
        """
        return self._execute_input_command(vm_index, f"tap {x} {y}", base_port)

    def swipe(
        self,
        vm_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
        base_port: int = 21503
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
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if swipe was successful, False otherwise.
        """
        return self._execute_input_command(
            vm_index,
            f"swipe {x1} {y1} {x2} {y2} {duration_ms}",
            base_port
        )

    def long_press(self, vm_index: int, x: int, y: int, duration_ms: int = 500, base_port: int = 21503) -> bool:
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
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if long press was successful, False otherwise.
        """
        return self._execute_input_command(
            vm_index,
            f"swipe {x} {y} {x} {y} {duration_ms}",
            base_port
        )

    def key_event(self, vm_index: int, key_code: int, base_port: int = 21503) -> bool:
        """
        Send a key event (e.g., back, home, menu).

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        key_code : int
            Android key code (e.g., 4 for back, 3 for home, 82 for menu).
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if key event was successful, False otherwise.
        """
        return self._execute_input_command(vm_index, f"keyevent {key_code}", base_port)

    def back(self, vm_index: int, base_port: int = 21503) -> bool:
        """
        Press the back button.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if back button press was successful, False otherwise.
        """
        return self.key_event(vm_index, 4, base_port)  # KEYCODE_BACK

    def home(self, vm_index: int, base_port: int = 21503) -> bool:
        """
        Press the home button.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if home button press was successful, False otherwise.
        """
        return self.key_event(vm_index, 3, base_port)  # KEYCODE_HOME

    def menu(self, vm_index: int, base_port: int = 21503) -> bool:
        """
        Press the menu button.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if menu button press was successful, False otherwise.
        """
        return self.key_event(vm_index, 82, base_port)  # KEYCODE_MENU

    def text(self, vm_index: int, text: str, base_port: int = 21503) -> bool:
        """
        Input text (type text on the screen).

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        text : str
            Text to input.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if text input was successful, False otherwise.
        """
        # Escape special characters and wrap in quotes
        escaped_text = text.replace(" ", "%s").replace("&", "\\&")
        return self._execute_input_command(vm_index, f"text {escaped_text}", base_port)

    def multi_touch(
        self,
        vm_index: int,
        points: List[Tuple[int, int]],
        base_port: int = 21503
    ) -> bool:
        """
        Perform a multi-touch gesture (pinch, zoom, etc.).

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        points : List[Tuple[int, int]]
            List of (x, y) coordinate tuples for touch points.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if multi-touch was successful, False otherwise.
        """
        if len(points) < 2:
            print("Multi-touch requires at least 2 points")
            return False

        # For multi-touch, we use sendevent or a combination of swipes
        # This is a simplified implementation using multiple taps
        port = self.get_adb_port(vm_index, base_port)
        try:
            # Use monkeyrunner-style approach or multiple simultaneous taps
            # Note: ADB input doesn't natively support multi-touch well,
            # so we'll simulate it with rapid sequential taps
            for x, y in points:
                self.tap(vm_index, x, y, base_port)
                time.sleep(0.05)  # Small delay between touches
            return True
        except Exception as e:
            print(f"Error performing multi-touch on VM {vm_index}: {e}")
            return False

    def drag(
        self,
        vm_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        steps: int = 10,
        base_port: int = 21503
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
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if drag was successful, False otherwise.
        """
        if steps < 2:
            return self.swipe(vm_index, x1, y1, x2, y2, 300, base_port)

        # Calculate step size
        dx = (x2 - x1) / steps
        dy = (y2 - y1) / steps
        duration_per_step = 300 // steps

        # Perform drag in steps
        current_x, current_y = x1, y1
        for i in range(steps - 1):
            next_x = int(x1 + dx * (i + 1))
            next_y = int(y1 + dy * (i + 1))
            self.swipe(vm_index, int(current_x), int(current_y), next_x, next_y, duration_per_step, base_port)
            current_x, current_y = next_x, next_y
            time.sleep(0.01)

        # Final step to exact end position
        return self.swipe(vm_index, int(current_x), int(current_y), x2, y2, duration_per_step, base_port)

    def pinch_zoom(
        self,
        vm_index: int,
        center_x: int,
        center_y: int,
        start_distance: int,
        end_distance: int,
        zoom_in: bool = True,
        base_port: int = 21503
    ) -> bool:
        """
        Perform a pinch-to-zoom gesture.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        center_x : int
            Center X coordinate of the zoom.
        center_y : int
            Center Y coordinate of the zoom.
        start_distance : int
            Starting distance between touch points.
        end_distance : int
            Ending distance between touch points.
        zoom_in : bool, optional
            True for zoom in, False for zoom out. Default is True.
        base_port : int, optional
            Base port number. Default is 21503.

        Returns
        -------
        bool
            True if pinch zoom was successful, False otherwise.
        """
        # Calculate two touch points
        if zoom_in:
            # Start close, end far
            start_dist = start_distance
            end_dist = end_distance
        else:
            # Start far, end close
            start_dist = end_distance
            end_dist = start_distance

        # Calculate points on a line (horizontal for simplicity)
        x1_start = center_x - start_dist // 2
        x1_end = center_x - end_dist // 2
        x2_start = center_x + start_dist // 2
        x2_end = center_x + end_dist // 2

        # Perform simultaneous swipes (simulated with sequential swipes)
        # Note: True multi-touch requires more complex implementation
        steps = 20
        duration = 500

        for i in range(steps):
            progress = i / steps
            x1 = int(x1_start + (x1_end - x1_start) * progress)
            x2 = int(x2_start + (x2_end - x2_start) * progress)

            # Simulate two-finger gesture with rapid taps
            self.tap(vm_index, x1, center_y, base_port)
            time.sleep(0.01)
            self.tap(vm_index, x2, center_y, base_port)
            time.sleep(duration / (steps * 1000))

        return True

