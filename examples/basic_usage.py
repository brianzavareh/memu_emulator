"""
Basic usage example for MEmu controller.

This example demonstrates basic operations: creating, starting, and controlling a VM.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to Python path to allow importing memu_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import MemuController
# #region agent log
log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "C", "location": "basic_usage.py:17", "message": "Controller initialized", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion

controller = MemuController()
vm_index = 0  # Your VM index

# #region agent log
vm_status = controller.get_vm_status(vm_index)
# Check screen state via ADB
screen_state = None
screen_size = None
try:
    screen_state_cmd = controller.execute_adb_command(vm_index, "dumpsys power | grep 'mScreenOn'")
    screen_state = screen_state_cmd
    screen_size = controller.get_screen_size(vm_index)
except Exception as e:
    screen_state = f"Error: {e}"
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "G", "location": "basic_usage.py:23", "message": "VM status and screen check", "data": {"vm_status": vm_status, "screen_state": screen_state, "screen_size": screen_size}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion

# #region agent log
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "basic_usage.py:27", "message": "ADB path used", "data": {"adb_path": controller.adb_manager.adb_path}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion

# Take screenshot
# #region agent log
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "basic_usage.py:31", "message": "Before take_screenshot_image", "data": {"vm_index": vm_index}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion
screenshot = controller.take_screenshot_image(vm_index)
# #region agent log
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "basic_usage.py:34", "message": "After take_screenshot_image", "data": {"screenshot_is_none": screenshot is None, "screenshot_type": type(screenshot).__name__ if screenshot else None}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion
if screenshot is None:
    # #region agent log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "basic_usage.py:37", "message": "Screenshot is None - exiting", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
    # #endregion
    print("Error: Screenshot is None. Check logs for details.")
    exit(1)
screenshot.save("game_screen.png")

# Tap at coordinates
controller.tap(vm_index, 500, 300)

# Find and tap a button
controller.tap_template(vm_index, "play_button.png", threshold=0.8)

# Swipe gesture
controller.swipe(vm_index, 100, 500, 900, 500, 500)