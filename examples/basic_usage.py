"""
Basic usage example for BlueStacks controller.

This example demonstrates opening the Daily Logic Puzzles app, taking a screenshot,
and extracting all text from the current window.
"""

import sys
import xml.etree.ElementTree as ET
import tempfile
import os
import subprocess
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

# # Find and launch the Daily Logic Puzzles app
# print("Searching for Daily Logic Puzzles app...")
# all_packages = controller.execute_adb_command(vm_index, "pm list packages")
# if not all_packages:
#     print("Error: Could not list packages.")
#     exit(1)

# # Search for packages containing "logic" or "puzzle" in Python
# package_name = None
# for line in all_packages.strip().split('\n'):
#     if line.startswith('package:'):
#         pkg = line.replace('package:', '').strip().lower()
#         if 'logic' in pkg or 'puzzle' in pkg:
#             package_name = line.replace('package:', '').strip()
#             print(f"Found matching package: {package_name}")
#             break

# if not package_name:
#     print("Error: Could not find Daily Logic Puzzles app. Please ensure it is installed.")
#     exit(1)

# print("Launching Daily Logic Puzzles app...")
# launch_result = controller.execute_adb_command(
#     vm_index,
#     f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
# )

# Take screenshot and extract text from current window (non-interactive)
print("\nTaking screenshot of current window (no interactions)...")
screenshot = controller.take_screenshot_image(vm_index, refresh_display=False)
if screenshot is None:
    print("Error: Screenshot is None. Check logs for details.")
    exit(1)

screenshot.save("game_screen.png")
print("Screenshot saved to game_screen.png")

# Extract all text from the current window using UI Automator
print("\nExtracting text from current window...")
device_dump_path = "/sdcard/ui_dump.xml"

# Dump UI hierarchy to device
dump_result = controller.execute_adb_command(vm_index, f"uiautomator dump {device_dump_path}")
if not dump_result or "ERROR" in dump_result.upper():
    print(f"Warning: UI dump command may have failed: {dump_result}")
    print("Attempting to continue anyway...")

# Pull the dump file to local machine
local_dump_path = os.path.join(tempfile.gettempdir(), f"ui_dump_{vm_index}.xml")
port = controller.config.get_adb_port(vm_index)
adb_path = controller.adb_manager.adb_path

pull_cmd = [adb_path, "-s", f"127.0.0.1:{port}", "pull", device_dump_path, local_dump_path]
pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=10)

if pull_result.returncode != 0:
    print(f"Error pulling UI dump file: {pull_result.stderr}")
    print("Text extraction failed.")
else:
    print(f"UI dump saved to {local_dump_path}")
    
    # Parse XML and extract all text
    try:
        tree = ET.parse(local_dump_path)
        root = tree.getroot()
        
        # Extract all text from nodes
        all_text = []
        for node in root.iter():
            # Get text attribute
            text = node.get('text', '').strip()
            if text:
                all_text.append(text)
            
            # Get content-desc attribute (often used for accessibility)
            content_desc = node.get('content-desc', '').strip()
            if content_desc and content_desc not in all_text:
                all_text.append(content_desc)
        
        # Print all found text
        print("\n" + "="*60)
        print("All text found in current window:")
        print("="*60)
        if all_text:
            for i, text in enumerate(all_text, 1):
                print(f"{i}. {text}")
        else:
            print("No text elements found in the UI hierarchy.")
        print("="*60)
        
        # Clean up local dump file
        try:
            os.remove(local_dump_path)
        except Exception:
            pass
        
        # Clean up device dump file
        controller.execute_adb_command(vm_index, f"rm {device_dump_path}")
        
    except ET.ParseError as e:
        print(f"Error parsing UI dump XML: {e}")
    except Exception as e:
        print(f"Error extracting text: {e}")