# BlueStacks Emulator Controller

A modular Python package for controlling BlueStacks emulators via ADB. This package provides a high-level, object-oriented interface for managing BlueStacks instances and performing ADB operations.

## Features

- **Modular Architecture**: Separated into distinct modules for instance detection, ADB operations, and configuration
- **Easy to Use**: High-level controller class that combines all functionality
- **Flexible Configuration**: Customizable settings for ADB ports and timeouts
- **Comprehensive Operations**: Detect running instances; execute ADB commands; install APKs
- **Status Monitoring**: Check instance status, running state, and ADB connectivity
- **Screenshot Capture**: Take screenshots of the emulator screen for analysis
- **Gesture Control**: Full gesture support (tap, swipe, long press, drag, pinch zoom)
- **Image Processing**: Template matching, color detection, and image analysis utilities
- **Game Automation**: Ready-to-use examples for automating games using screenshots and gestures

## Prerequisites

- Python 3.7 or higher
- BlueStacks installed on your system
- ADB (Android Debug Bridge) - **Included in this repository** in the `platform-tools/` folder, or can be installed separately

## Installation

### 1. Create Virtual Environment

**Windows:**
```bash
setup_env.bat
```

**Linux/Mac:**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv android_env

# Activate virtual environment
# Windows:
android_env\Scripts\activate
# Linux/Mac:
source android_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. ADB Configuration

**ADB is included in this repository** in the `platform-tools/` folder. The package will automatically detect and use it. No additional configuration is needed.

If you prefer to use a system-wide ADB installation:
- Make sure ADB is in your system PATH, or
- Configure the ADB path manually (see Configuration section below)

### 3. Verify BlueStacks Installation

Make sure BlueStacks is installed and running. BlueStacks instances must be created and started manually through the BlueStacks application. The package will detect running instances via ADB.

## Project Structure

```
android_emulator/
├── android_controller/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── controller.py        # Main controller class
│   ├── vm_manager.py        # VM lifecycle management
│   ├── adb_manager.py       # ADB operations
│   ├── input_manager.py     # Input and gesture control
│   └── image_utils.py       # Image processing utilities
├── platform-tools/          # ADB tools (included)
│   ├── adb.exe              # Android Debug Bridge
│   └── ...                  # ADB support files
├── examples/                 # Example scripts
│   ├── basic_usage.py       # Basic operations example
│   ├── advanced_usage.py    # Advanced operations example
│   ├── interactive_control.py  # Interactive CLI example
│   └── game_automation.py   # Game automation example
├── tests/                    # Test scripts
│   └── test_screenshot_gestures.py  # Screenshot and gesture tests
├── requirements.txt         # Python dependencies
├── setup_env.bat            # Windows setup script
├── setup_env.sh             # Linux/Mac setup script
├── README.md                # This file
└── QUICK_START_GAME_AUTOMATION.md  # Game automation quick start
```

## Quick Start

### Basic Usage

```python
from android_controller import BlueStacksController

# Create controller
controller = BlueStacksController()

# List existing BlueStacks instances (detected via ADB)
instances = controller.list_vms()
print(f"Found {len(instances)} BlueStacks instances")

# Connect to an instance (assuming instance 0 is running)
vm_index = 0
if not controller.get_vm_status(vm_index)['adb_connected']:
    controller.connect_adb(vm_index)

# Execute ADB commands
android_version = controller.execute_adb_command(
    vm_index, 
    "getprop ro.build.version.release"
)
print(f"Android Version: {android_version}")

# Disconnect ADB
controller.disconnect_adb(vm_index)
```

### Custom Configuration

```python
from android_controller import BlueStacksController, BlueStacksConfig

# Create custom configuration
config = BlueStacksConfig(
    timeout=90,
    adb_port_base=5555  # BlueStacks default port
)

# Create controller with custom config
controller = BlueStacksController(config=config)
```

## Usage Examples

### Example 1: Basic VM Operations

See `examples/basic_usage.py` for a complete example of:
- Listing VMs
- Creating and starting VMs
- Getting VM status
- Executing ADB commands
- Stopping VMs

### Example 2: Advanced Operations

See `examples/advanced_usage.py` for:
- Custom configuration
- Multiple VM management
- System information retrieval
- APK installation (commented example)

### Example 3: Interactive Control

Run `examples/interactive_control.py` for an interactive command-line interface:

```bash
python examples/interactive_control.py
```

## API Reference

### BlueStacksController

Main controller class that provides unified interface for all operations.

#### Methods

- `list_vms()` - List all available BlueStacks instances (detected via ADB)
- `create_and_start_vm(vm_name=None)` - Note: BlueStacks instances must be created manually
- `start_vm(vm_index, connect_adb=True)` - Check if instance is running and connect ADB
- `stop_vm(vm_index, disconnect_adb=True)` - Disconnect ADB (instances must be stopped manually)
- `delete_vm(vm_index, force_stop=True)` - Note: BlueStacks instances must be deleted manually
- `get_vm_status(vm_index)` - Get comprehensive instance status
- `connect_adb(vm_index)` - Connect to instance via ADB
- `disconnect_adb(vm_index)` - Disconnect from instance via ADB
- `execute_adb_command(vm_index, command)` - Execute ADB shell command
- `install_apk(vm_index, apk_path)` - Install APK on instance
- `set_active_vm(vm_index)` - Set active instance for operations
- `get_active_vm()` - Get currently active instance index
- `take_screenshot(vm_index, save_path=None)` - Take screenshot and save to file
- `take_screenshot_image(vm_index)` - Take screenshot as PIL Image
- `get_screen_size(vm_index)` - Get screen dimensions (width, height)
- `tap(vm_index, x, y)` - Tap at coordinates
- `swipe(vm_index, x1, y1, x2, y2, duration_ms=300)` - Swipe gesture
- `long_press(vm_index, x, y, duration_ms=500)` - Long press gesture
- `drag(vm_index, x1, y1, x2, y2, steps=10)` - Drag gesture
- `back(vm_index)` - Press back button
- `home(vm_index)` - Press home button
- `find_template_in_screenshot(vm_index, template_path, threshold=0.8)` - Find template in screenshot
- `tap_template(vm_index, template_path, threshold=0.8)` - Find and tap template

### BlueStacksConfig

Configuration class for customizing controller behavior.

#### Attributes

- `adb_path` - Path to adb.exe (None for auto-detect, checks local `platform-tools/` folder first)
- `timeout` - Default timeout for operations (seconds)
- `adb_port_base` - Base port for ADB connections (default: 5555 for BlueStacks)

#### ADB Auto-Detection

The package automatically detects ADB in the following order:
1. **Local `platform-tools/` folder** (included in repository) - **Highest priority**
2. System PATH
3. BlueStacks installation directory
4. Android SDK platform-tools directory

You can also manually specify the ADB path:

```python
from android_controller import BlueStacksController, BlueStacksConfig

config = BlueStacksConfig(
    adb_path="C:/path/to/adb.exe"  # Custom ADB path
)
controller = BlueStacksController(config=config)
```

### VMManager

Low-level VM management operations.

### ADBManager

Low-level ADB operations.

### InputManager

Low-level input and gesture operations.

### ImageProcessor

Image processing utilities for screenshot analysis.

## Screenshot and Gesture Control

### Taking Screenshots

```python
# Take screenshot and save to file
screenshot_path = controller.take_screenshot(vm_index, "screenshot.png")

# Take screenshot as PIL Image for processing
screenshot = controller.take_screenshot_image(vm_index)
if screenshot:
    screenshot.save("my_screenshot.png")
    print(f"Screenshot size: {screenshot.size}")

# Get screen dimensions
screen_size = controller.get_screen_size(vm_index)
width, height = screen_size
```

### Gesture Control

```python
# Tap at coordinates
controller.tap(vm_index, x=500, y=300)

# Swipe gesture
controller.swipe(vm_index, x1=100, y1=500, x2=900, y2=500, duration_ms=500)

# Long press
controller.long_press(vm_index, x=500, y=300, duration_ms=1000)

# Drag with smooth movement
controller.drag(vm_index, x1=100, y1=100, x2=800, y2=800, steps=20)

# Navigation buttons
controller.back(vm_index)
controller.home(vm_index)
```

### Image Processing

```python
from android_controller import ImageProcessor

# Take screenshot
screenshot = controller.take_screenshot_image(vm_index)

# Find template in screenshot
bbox = ImageProcessor.find_template(screenshot, "button_template.png", threshold=0.8)
if bbox:
    x, y, w, h = bbox
    center = ImageProcessor.get_center(bbox)
    controller.tap(vm_index, center[0], center[1])

# Find regions by color
red_regions = ImageProcessor.find_color_region(screenshot, (255, 0, 0), tolerance=30)
for region in red_regions:
    center = ImageProcessor.get_center(region)
    controller.tap(vm_index, center[0], center[1])
```

### Game Automation Example

```python
# Find and tap on a template
if controller.tap_template(vm_index, "play_button.png", threshold=0.8):
    print("Play button found and tapped!")

# Automation loop
for i in range(10):
    # Take screenshot
    screenshot = controller.take_screenshot_image(vm_index)
    
    # Process screenshot and make decisions
    # ... your game logic here ...
    
    # Perform actions
    controller.tap(vm_index, x, y)
    time.sleep(2)
```

## Common ADB Commands

Here are some useful ADB commands you can execute:

```python
# Get Android version
controller.execute_adb_command(vm_index, "getprop ro.build.version.release")

# Get device model
controller.execute_adb_command(vm_index, "getprop ro.product.model")

# List installed packages
controller.execute_adb_command(vm_index, "pm list packages")

# Get screen resolution
controller.execute_adb_command(vm_index, "wm size")

# Get battery level
controller.execute_adb_command(vm_index, "dumpsys battery | grep level")

# Launch an app
controller.execute_adb_command(
    vm_index, 
    "monkey -p com.android.settings -c android.intent.category.LAUNCHER 1"
)
```

## Troubleshooting

### BlueStacks Instance Not Detected

- Ensure BlueStacks is installed and running
- Make sure the BlueStacks instance is fully booted before connecting
- Verify ADB can see the instance: `adb devices`
- Try manually connecting: `adb connect 127.0.0.1:5555`

### ADB Connection Fails

- Ensure the BlueStacks instance is fully booted before connecting
- **ADB is included in the repository** - the package will auto-detect it from `platform-tools/` folder
- If using system ADB, check that ADB is installed and in PATH
- Verify the ADB port (default: 5555 for first instance)
- Try manually connecting: `adb connect 127.0.0.1:5555`
- Check that the `platform-tools/adb.exe` file exists in the project root
- Enable ADB in BlueStacks settings if needed

### Instance Not Starting

- BlueStacks instances must be started manually through the BlueStacks application
- Check BlueStacks is running
- Verify system resources (RAM, CPU)
- Check BlueStacks logs for errors

## Contributing

This is a modular package designed for easy extension. You can:

1. Add new modules in `android_controller/`
2. Extend existing classes with new methods
3. Create custom configuration options
4. Add new example scripts

## License

This project is provided as-is for controlling Android emulators programmatically.

## Acknowledgments

- Designed for use with BlueStacks emulator
- Uses ADB (Android Debug Bridge) for communication

