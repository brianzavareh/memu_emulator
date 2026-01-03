# Quick Start Guide: Game Automation

This guide will help you get started with automating games in MEmu emulator using screenshot analysis and gesture control.

## Prerequisites

1. MEmu Play installed and running
2. At least one VM created in MEmu
3. Python virtual environment set up (see main README)

## Installation

```bash
# Activate virtual environment
memu_env\Scripts\activate  # Windows
# or
source memu_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Basic Game Automation Workflow

### 1. Connect to Your VM

```python
from memu_controller import MemuController

controller = MemuController()

# List VMs and select one
vms = controller.list_vms()
vm_index = vms[0].get('index')  # Use first VM

# Start VM if not running
if not controller.get_vm_status(vm_index)['running']:
    controller.start_vm(vm_index)
    time.sleep(10)  # Wait for boot

# Connect ADB
controller.connect_adb(vm_index)
```

### 2. Take Screenshots

```python
import time

# Take screenshot and save
screenshot_path = controller.take_screenshot(vm_index, "current_screen.png")

# Or get as PIL Image for processing
screenshot = controller.take_screenshot_image(vm_index)
if screenshot:
    screenshot.save("screenshot.png")
```

### 3. Perform Gestures

```python
# Get screen size
width, height = controller.get_screen_size(vm_index)

# Tap at center
controller.tap(vm_index, width // 2, height // 2)

# Swipe left to right
controller.swipe(vm_index, 100, height // 2, width - 100, height // 2, 500)

# Long press
controller.long_press(vm_index, width // 2, height // 2, 1000)
```

### 4. Template Matching (Find and Tap Buttons)

```python
# Save a template image of a button you want to find
# Then use template matching:

bbox = controller.find_template_in_screenshot(
    vm_index, 
    "play_button.png",  # Path to template image
    threshold=0.8
)

if bbox:
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    controller.tap(vm_index, center_x, center_y)
```

### 5. Simple Automation Loop

```python
import time

for i in range(10):
    # Take screenshot
    screenshot = controller.take_screenshot_image(vm_index)
    
    # Analyze screenshot (add your logic here)
    # For example, find a button and tap it
    
    # Perform action
    controller.tap(vm_index, x, y)
    
    # Wait before next iteration
    time.sleep(2)
```

## Complete Example: Simple Game Bot

```python
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import MemuController, ImageProcessor

def game_bot(controller: MemuController, vm_index: int):
    """Simple game automation bot."""
    
    # Get screen size
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("Failed to get screen size")
        return
    
    width, height = screen_size
    print(f"Screen size: {width}x{height}")
    
    # Main loop
    for iteration in range(20):
        print(f"\nIteration {iteration + 1}")
        
        # Take screenshot
        screenshot = controller.take_screenshot_image(vm_index)
        if not screenshot:
            print("Failed to take screenshot")
            continue
        
        # Save screenshot for debugging
        screenshot.save(f"game_screenshot_{iteration+1}.png")
        
        # Example: Tap at different locations
        # Replace with your game logic
        tap_x = width // 2 + (iteration % 3 - 1) * 100
        tap_y = height // 2
        
        print(f"Tapping at: ({tap_x}, {tap_y})")
        controller.tap(vm_index, tap_x, tap_y)
        
        # Wait for game to respond
        time.sleep(3)

if __name__ == "__main__":
    controller = MemuController()
    
    # Get VM
    vms = controller.list_vms()
    if not vms:
        print("No VMs found")
        exit(1)
    
    vm_index = vms[0].get('index')
    
    # Ensure VM is running
    status = controller.get_vm_status(vm_index)
    if not status['running']:
        print("Starting VM...")
        controller.start_vm(vm_index)
        time.sleep(10)
    
    if not status['adb_connected']:
        print("Connecting ADB...")
        controller.connect_adb(vm_index)
        time.sleep(2)
    
    # Run bot
    try:
        game_bot(controller, vm_index)
    except KeyboardInterrupt:
        print("\nBot stopped by user")
```

## Advanced: Using Template Matching

1. **Capture Template Images**:
   - Take a screenshot when you see the button/element you want to find
   - Crop the image to just the button
   - Save it as a template (e.g., `play_button.png`)

2. **Use Template Matching**:
   ```python
   # Find and tap play button
   if controller.tap_template(vm_index, "play_button.png", threshold=0.8):
       print("Play button tapped!")
   else:
       print("Play button not found")
   ```

## Advanced: Color-Based Detection

```python
from memu_controller import ImageProcessor

# Take screenshot
screenshot = controller.take_screenshot_image(vm_index)

# Find all red regions (adjust color for your game)
red_regions = ImageProcessor.find_color_region(
    screenshot, 
    (255, 0, 0),  # RGB color
    tolerance=30
)

# Tap on each red region found
for region in red_regions:
    center = ImageProcessor.get_center(region)
    controller.tap(vm_index, center[0], center[1])
    time.sleep(0.5)
```

## Tips for Game Automation

1. **Wait Times**: Always add appropriate delays between actions to let the game respond
2. **Error Handling**: Check if screenshots are captured successfully
3. **Template Images**: Use high-quality template images for better matching
4. **Threshold Tuning**: Adjust template matching threshold (0.7-0.9) based on your needs
5. **Screen Resolution**: Be aware that coordinates change with different screen sizes
6. **Debugging**: Save screenshots frequently to debug your automation logic

## Testing Your Automation

Run the test suite to verify everything works:

```bash
python tests/test_screenshot_gestures.py
```

## Next Steps

- Check `examples/game_automation.py` for more examples
- Explore `memu_controller/image_utils.py` for advanced image processing
- Customize the automation logic for your specific game

