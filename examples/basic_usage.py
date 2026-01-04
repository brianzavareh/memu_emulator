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
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

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
    
    # Parse XML and extract all text with coordinates
    try:
        tree = ET.parse(local_dump_path)
        root = tree.getroot()
        
        def parse_bounds(bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
            """
            Parse bounds string in format [x1,y1][x2,y2] to (x1, y1, x2, y2).
            
            Parameters
            ----------
            bounds_str : str
                Bounds string from UI Automator XML.
            
            Returns
            -------
            Optional[Tuple[int, int, int, int]]
                Tuple of (x1, y1, x2, y2) or None if parsing fails.
            """
            if not bounds_str:
                return None
            # Match pattern [x1,y1][x2,y2]
            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
            if match:
                return tuple(map(int, match.groups()))
            return None
        
        def get_center(bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
            """
            Calculate center point of bounds rectangle.
            
            Parameters
            ----------
            bounds : Tuple[int, int, int, int]
                Bounds as (x1, y1, x2, y2).
            
            Returns
            -------
            Tuple[int, int]
                Center point as (x, y).
            """
            x1, y1, x2, y2 = bounds
            return ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Extract all text elements with their coordinates
        text_elements: List[Dict[str, Any]] = []
        seen_texts = set()  # To avoid duplicates
        
        for node in root.iter():
            # Get text attribute
            text = node.get('text', '').strip()
            bounds_str = node.get('bounds', '')
            bounds = parse_bounds(bounds_str)
            
            if text and bounds:
                # Create unique key to avoid exact duplicates
                text_key = f"{text}_{bounds}"
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    center = get_center(bounds)
                    text_elements.append({
                        'text': text,
                        'bounds': bounds,
                        'center': center,
                        'source': 'text'
                    })
            
            # Get content-desc attribute (often used for accessibility)
            content_desc = node.get('content-desc', '').strip()
            if content_desc and bounds:
                # Create unique key to avoid exact duplicates
                desc_key = f"{content_desc}_{bounds}"
                if desc_key not in seen_texts:
                    seen_texts.add(desc_key)
                    center = get_center(bounds)
                    text_elements.append({
                        'text': content_desc,
                        'bounds': bounds,
                        'center': center,
                        'source': 'content-desc'
                    })
        
        # Print all found text with coordinates
        print("\n" + "="*80)
        print("All text found in current window (with coordinates):")
        print("="*80)
        if text_elements:
            for i, elem in enumerate(text_elements, 1):
                x1, y1, x2, y2 = elem['bounds']
                center_x, center_y = elem['center']
                print(f"\n{i}. Text: \"{elem['text']}\"")
                print(f"   Bounds: [{x1},{y1}][{x2},{y2}] (width: {x2-x1}, height: {y2-y1})")
                print(f"   Center: ({center_x}, {center_y})")
                print(f"   Source: {elem['source']}")
                print(f"   Click command: controller.tap({vm_index}, {center_x}, {center_y})")
            
            # Summary section with all coordinates in compact format
            print("\n" + "="*80)
            print("SUMMARY - Quick Reference for Clicking:")
            print("="*80)
            for i, elem in enumerate(text_elements, 1):
                center_x, center_y = elem['center']
                print(f"{i:3d}. \"{elem['text'][:40]:<40}\" -> tap({vm_index}, {center_x:4d}, {center_y:4d})")
            print("="*80)
        else:
            print("No text elements found in the UI hierarchy.")
            print("="*80)
        
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

# Find and tap on "Color Flood" text using the modular function
print("\n" + "="*80)
print("Finding and tapping on 'Color Flood' text...")
print("="*80)
if controller.find_and_tap_text(vm_index, "Color Flood"):
    print("Successfully tapped on 'Color Flood'!")
else:
    print("Could not find 'Color Flood' text in the UI.")
    # Try to find coordinates for debugging
    coords = controller.find_text_coordinates(vm_index, "Color Flood")
    if coords:
        print(f"Found 'Color Flood' at center: {coords['center']}")
        print(f"Bounds: {coords['bounds']}")
        print("Attempting manual tap...")
        controller.tap(vm_index, coords['center'][0], coords['center'][1])
    else:
        print("Text 'Color Flood' not found in UI hierarchy.")