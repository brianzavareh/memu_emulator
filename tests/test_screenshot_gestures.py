"""
Test script for screenshot and gesture functionality.

This script tests the screenshot capture and gesture control capabilities.
"""

import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import MemuController, ImageProcessor


def test_screenshot_capture(controller: MemuController, vm_index: int) -> bool:
    """Test screenshot capture functionality."""
    print("\n[TEST] Screenshot Capture")
    print("-" * 50)
    
    # Test 1: Take screenshot as bytes
    print("Test 1: Taking screenshot as bytes...")
    screenshot_bytes = controller.adb_manager.take_screenshot_bytes(
        vm_index, controller.config.adb_port_base
    )
    if screenshot_bytes:
        print(f"✓ Screenshot captured: {len(screenshot_bytes)} bytes")
    else:
        print("✗ Failed to capture screenshot")
        return False
    
    # Test 2: Take screenshot as PIL Image
    print("\nTest 2: Taking screenshot as PIL Image...")
    screenshot_image = controller.take_screenshot_image(vm_index)
    if screenshot_image:
        print(f"✓ Screenshot captured as PIL Image: {screenshot_image.size}")
    else:
        print("✗ Failed to capture screenshot as image")
        return False
    
    # Test 3: Save screenshot to file
    print("\nTest 3: Saving screenshot to file...")
    save_path = "test_screenshot.png"
    result_path = controller.take_screenshot(vm_index, save_path)
    if result_path and Path(result_path).exists():
        print(f"✓ Screenshot saved to: {result_path}")
        Path(result_path).unlink()  # Clean up
    else:
        print("✗ Failed to save screenshot")
        return False
    
    print("\n✓ All screenshot tests passed!")
    return True


def test_screen_size(controller: MemuController, vm_index: int) -> bool:
    """Test screen size detection."""
    print("\n[TEST] Screen Size Detection")
    print("-" * 50)
    
    screen_size = controller.get_screen_size(vm_index)
    if screen_size:
        width, height = screen_size
        print(f"✓ Screen size detected: {width}x{height}")
        return True
    else:
        print("✗ Failed to detect screen size")
        return False


def test_tap_gesture(controller: MemuController, vm_index: int) -> bool:
    """Test tap gesture."""
    print("\n[TEST] Tap Gesture")
    print("-" * 50)
    
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("✗ Cannot test tap - screen size unknown")
        return False
    
    width, height = screen_size
    center_x, center_y = width // 2, height // 2
    
    print(f"Tapping at center: ({center_x}, {center_y})")
    result = controller.tap(vm_index, center_x, center_y)
    
    if result:
        print("✓ Tap gesture executed successfully")
        time.sleep(0.5)
        return True
    else:
        print("✗ Tap gesture failed")
        return False


def test_swipe_gesture(controller: MemuController, vm_index: int) -> bool:
    """Test swipe gesture."""
    print("\n[TEST] Swipe Gesture")
    print("-" * 50)
    
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("✗ Cannot test swipe - screen size unknown")
        return False
    
    width, height = screen_size
    
    # Swipe from left to right
    print("Swiping left to right...")
    result = controller.swipe(
        vm_index,
        width // 4, height // 2,
        width * 3 // 4, height // 2,
        500
    )
    
    if result:
        print("✓ Swipe gesture executed successfully")
        time.sleep(0.5)
        return True
    else:
        print("✗ Swipe gesture failed")
        return False


def test_long_press(controller: MemuController, vm_index: int) -> bool:
    """Test long press gesture."""
    print("\n[TEST] Long Press Gesture")
    print("-" * 50)
    
    screen_size = controller.get_screen_size(vm_index)
    if not screen_size:
        print("✗ Cannot test long press - screen size unknown")
        return False
    
    width, height = screen_size
    center_x, center_y = width // 2, height // 2
    
    print(f"Long pressing at: ({center_x}, {center_y})")
    result = controller.long_press(vm_index, center_x, center_y, 500)
    
    if result:
        print("✓ Long press gesture executed successfully")
        time.sleep(0.5)
        return True
    else:
        print("✗ Long press gesture failed")
        return False


def test_navigation_buttons(controller: MemuController, vm_index: int) -> bool:
    """Test navigation button presses."""
    print("\n[TEST] Navigation Buttons")
    print("-" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test home button
    print("Testing home button...")
    if controller.home(vm_index):
        print("✓ Home button pressed")
        tests_passed += 1
        time.sleep(1)
    else:
        print("✗ Home button failed")
    
    # Test back button
    print("Testing back button...")
    if controller.back(vm_index):
        print("✓ Back button pressed")
        tests_passed += 1
        time.sleep(1)
    else:
        print("✗ Back button failed")
    
    # Test menu button
    print("Testing menu button...")
    if controller.input_manager.menu(vm_index, controller.config.adb_port_base):
        print("✓ Menu button pressed")
        tests_passed += 1
        time.sleep(1)
    else:
        print("✗ Menu button failed")
    
    print(f"\n✓ {tests_passed}/{total_tests} navigation button tests passed")
    return tests_passed == total_tests


def test_image_processing(controller: MemuController, vm_index: int) -> bool:
    """Test image processing utilities."""
    print("\n[TEST] Image Processing")
    print("-" * 50)
    
    # Take screenshot
    screenshot = controller.take_screenshot_image(vm_index)
    if not screenshot:
        print("✗ Cannot test image processing - screenshot failed")
        return False
    
    print(f"✓ Screenshot loaded: {screenshot.size}")
    
    # Test image conversion
    print("Testing PIL to OpenCV conversion...")
    cv2_image = ImageProcessor.pil_to_cv2(screenshot)
    if cv2_image is not None:
        print(f"✓ Converted to OpenCV format: {cv2_image.shape}")
    else:
        print("✗ Conversion failed")
        return False
    
    # Test conversion back
    print("Testing OpenCV to PIL conversion...")
    pil_image = ImageProcessor.cv2_to_pil(cv2_image)
    if pil_image is not None:
        print(f"✓ Converted back to PIL: {pil_image.size}")
    else:
        print("✗ Conversion failed")
        return False
    
    # Test color region detection (just test the function, may not find anything)
    print("Testing color region detection...")
    regions = ImageProcessor.find_color_region(screenshot, (255, 255, 255), tolerance=50)
    print(f"✓ Found {len(regions)} color regions (test may find 0, that's OK)")
    
    print("\n✓ All image processing tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MEmu Screenshot and Gesture Test Suite")
    print("=" * 60)
    
    controller = MemuController()
    
    # List VMs
    vms = controller.list_vms()
    if not vms:
        print("\n✗ No VMs found. Please create a VM first.")
        return
    
    print(f"\nFound {len(vms)} VM(s)")
    vm_index = vms[0].get('index')
    print(f"Using VM {vm_index} for testing\n")
    
    # Check VM status
    status = controller.get_vm_status(vm_index)
    if not status['running']:
        print(f"VM {vm_index} is not running. Starting it...")
        controller.start_vm(vm_index)
        print("Waiting for VM to boot...")
        time.sleep(10)
    
    if not status['adb_connected']:
        print("Connecting via ADB...")
        controller.connect_adb(vm_index)
        time.sleep(2)
    
    # Run tests
    tests = [
        ("Screenshot Capture", test_screenshot_capture),
        ("Screen Size", test_screen_size),
        ("Tap Gesture", test_tap_gesture),
        ("Swipe Gesture", test_swipe_gesture),
        ("Long Press", test_long_press),
        ("Navigation Buttons", test_navigation_buttons),
        ("Image Processing", test_image_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(controller, vm_index)
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()

