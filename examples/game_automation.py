"""
Game automation example using screenshot and gesture control.

This example demonstrates how to use screenshot analysis and gesture control
to automate game interactions.
"""

import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import BlueStacksController, ImageProcessor
from PIL import Image


def example_basic_screenshot_and_tap(controller: BlueStacksController, vm_index: int):
    """Example: Take screenshot and perform basic tap."""
    print("\n=== Basic Screenshot and Tap ===")
    
    # Take screenshot
    print("Taking screenshot...")
    screenshot = controller.take_screenshot_image(vm_index)
    if screenshot:
        print(f"Screenshot captured: {screenshot.size}")
        
        # Save screenshot
        screenshot.save("screenshot_example.png")
        print("Screenshot saved as 'screenshot_example.png'")
        
        # Get screen size
        screen_size = controller.get_screen_size(vm_index)
        if screen_size:
            print(f"Screen size: {screen_size[0]}x{screen_size[1]}")
            
            # Tap at center of screen
            center_x = screen_size[0] // 2
            center_y = screen_size[1] // 2
            print(f"Tapping at center: ({center_x}, {center_y})")
            controller.tap(vm_index, center_x, center_y)
            time.sleep(1)
    else:
        print("Failed to take screenshot")


def example_swipe_gestures(controller: BlueStacksController, vm_index: int):
    """Example: Perform various swipe gestures."""
    print("\n=== Swipe Gestures ===")
    
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("Failed to get screen size")
        return
    
    width, height = screen_size
    
    # Swipe from left to right (like swiping pages)
    print("Swiping left to right...")
    controller.swipe(vm_index, width // 4, height // 2, width * 3 // 4, height // 2, 500)
    time.sleep(1)
    
    # Swipe from right to left
    print("Swiping right to left...")
    controller.swipe(vm_index, width * 3 // 4, height // 2, width // 4, height // 2, 500)
    time.sleep(1)
    
    # Swipe up (scroll down)
    print("Swiping up (scroll down)...")
    controller.swipe(vm_index, width // 2, height * 3 // 4, width // 2, height // 4, 500)
    time.sleep(1)
    
    # Swipe down (scroll up)
    print("Swiping down (scroll up)...")
    controller.swipe(vm_index, width // 2, height // 4, width // 2, height * 3 // 4, 500)
    time.sleep(1)


def example_template_matching(controller: BlueStacksController, vm_index: int, template_path: str):
    """Example: Find template in screenshot and tap on it."""
    print("\n=== Template Matching ===")
    
    if not Path(template_path).exists():
        print(f"Template file not found: {template_path}")
        print("Skipping template matching example")
        return
    
    print(f"Looking for template: {template_path}")
    bbox = controller.find_template_in_screenshot(vm_index, template_path, threshold=0.8)
    
    if bbox:
        x, y, w, h = bbox
        print(f"Template found at: ({x}, {y}) size: {w}x{h}")
        
        # Get center and tap
        center = ImageProcessor.get_center(bbox)
        print(f"Tapping at center: {center}")
        controller.tap(vm_index, center[0], center[1])
    else:
        print("Template not found in screenshot")


def example_color_detection(controller: BlueStacksController, vm_index: int):
    """Example: Find regions by color and interact with them."""
    print("\n=== Color Detection ===")
    
    screenshot = controller.take_screenshot_image(vm_index)
    if not screenshot:
        print("Failed to take screenshot")
        return
    
    # Example: Find red regions (adjust color as needed)
    # This is just an example - adjust the color based on your game
    target_color = (255, 0, 0)  # Red in RGB
    regions = ImageProcessor.find_color_region(screenshot, target_color, tolerance=30)
    
    print(f"Found {len(regions)} regions matching color {target_color}")
    
    if regions:
        # Tap on the first region found
        center = ImageProcessor.get_center(regions[0])
        print(f"Tapping on first region at: {center}")
        controller.tap(vm_index, center[0], center[1])
    else:
        print("No matching color regions found")


def example_game_automation_loop(controller: BlueStacksController, vm_index: int, iterations: int = 5):
    """Example: Simple automation loop for repetitive tasks."""
    print("\n=== Game Automation Loop ===")
    
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("Failed to get screen size")
        return
    
    width, height = screen_size
    
    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        
        # Take screenshot
        screenshot = controller.take_screenshot_image(vm_index)
        if screenshot:
            # Save screenshot for analysis
            screenshot.save(f"automation_screenshot_{i+1}.png")
        
        # Example: Tap at a specific location (adjust coordinates for your game)
        # This is just an example - replace with your game's button coordinates
        tap_x = width // 2
        tap_y = height // 2
        print(f"Tapping at: ({tap_x}, {tap_y})")
        controller.tap(vm_index, tap_x, tap_y)
        
        # Wait before next iteration
        time.sleep(2)
    
    print("\nAutomation loop completed")


def main():
    """Main function demonstrating game automation capabilities."""
    print("=== BlueStacks Game Automation Example ===")
    
    # Initialize controller
    controller = BlueStacksController()
    
    # List BlueStacks instances
    print("\nListing BlueStacks instances...")
    instances = controller.list_vms()
    if not instances:
        print("No BlueStacks instances found. Please start BlueStacks first.")
        return
    
    print(f"Found {len(instances)} instance(s):")
    for instance in instances:
        print(f"  - Instance {instance.get('index')}: {instance.get('name', 'Unknown')}")
    
    # Get instance index (use first available or ask user)
    if instances:
        vm_index = instances[0].get('index')
        print(f"\nUsing instance {vm_index}")
        
        # Check if instance is running
        status = controller.get_vm_status(vm_index)
        if not status['running']:
            print(f"Instance {vm_index} is not running. Please start it manually in BlueStacks.")
            return
        
        # Connect ADB if not connected
        if not status['adb_connected']:
            print("Connecting via ADB...")
            controller.connect_adb(vm_index)
            time.sleep(2)
        
        # Run examples
        try:
            # Basic screenshot and tap
            example_basic_screenshot_and_tap(controller, vm_index)
            time.sleep(2)
            
            # Swipe gestures
            example_swipe_gestures(controller, vm_index)
            time.sleep(2)
            
            # Template matching (if template file exists)
            # Uncomment and provide path to test:
            # example_template_matching(controller, vm_index, "path/to/template.png")
            
            # Color detection
            example_color_detection(controller, vm_index)
            time.sleep(2)
            
            # Automation loop
            response = input("\nRun automation loop? (y/n): ")
            if response.lower() == 'y':
                iterations = input("Number of iterations (default 5): ")
                iterations = int(iterations) if iterations.isdigit() else 5
                example_game_automation_loop(controller, vm_index, iterations)
            
        except KeyboardInterrupt:
            print("\n\nAutomation interrupted by user")
        except Exception as e:
            print(f"\nError during automation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No VMs available")


if __name__ == "__main__":
    main()

