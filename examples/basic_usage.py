"""
Basic usage example for BlueStacks controller.

This example demonstrates opening the Daily Logic Puzzles app and taking a screenshot.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing memu_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import BlueStacksController

controller = BlueStacksController()
vm_index = 0  # Your BlueStacks instance index

vm_status = controller.get_vm_status(vm_index)

# Ensure BlueStacks instance is running and ADB is connected
if not vm_status['running']:
    print(f"BlueStacks instance {vm_index} is not running. Please start it manually in BlueStacks.")
    exit(1)

if not vm_status['adb_connected']:
    print("Connecting to VM via ADB...")
    if not controller.connect_adb(vm_index):
        print("Failed to connect via ADB. Please ensure VM is running.")
        exit(1)
    print("ADB connected successfully.")

# Find and launch the Daily Logic Puzzles app
print("Searching for Daily Logic Puzzles app...")
all_packages = controller.execute_adb_command(vm_index, "pm list packages")
if not all_packages:
    print("Error: Could not list packages.")
    exit(1)

# Search for packages containing "logic" or "puzzle" in Python
package_name = None
for line in all_packages.strip().split('\n'):
    if line.startswith('package:'):
        pkg = line.replace('package:', '').strip().lower()
        if 'logic' in pkg or 'puzzle' in pkg:
            package_name = line.replace('package:', '').strip()
            print(f"Found matching package: {package_name}")
            break

if not package_name:
    print("Error: Could not find Daily Logic Puzzles app. Please ensure it is installed.")
    exit(1)

print("Launching Daily Logic Puzzles app...")
launch_result = controller.execute_adb_command(
    vm_index,
    f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
)

# Take screenshot immediately after launching app

# Take screenshot of the app (uses Method 2: WAKEUP refresh)
print("\nTaking screenshot of Daily Logic Puzzles app...")
screenshot = controller.take_screenshot_image(vm_index)
if screenshot is None:
    print("Error: Screenshot is None. Check logs for details.")
    exit(1)

screenshot.save("game_screen.png")
print("Screenshot saved to game_screen.png")