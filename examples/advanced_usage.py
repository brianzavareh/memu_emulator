"""
Advanced usage example for BlueStacks controller.

This example demonstrates advanced operations including custom configuration,
multiple instance management, and APK installation.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing memu_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import BlueStacksController, BlueStacksConfig
import time


def main():
    """Demonstrate advanced BlueStacks controller usage."""
    # Create custom configuration
    config = BlueStacksConfig(
        timeout=90,
        adb_port_base=5555
    )
    
    controller = BlueStacksController(config=config)
    
    # List all BlueStacks instances and their status
    print("=== BlueStacks Instance Status Overview ===")
    instances = controller.list_vms()
    if not instances:
        print("No BlueStacks instances detected. Please start BlueStacks first.")
        return
    
    for instance in instances:
        vm_index = instance.get('index')
        status = controller.get_vm_status(vm_index)
        print(f"\nInstance {vm_index} ({instance.get('name', 'Unknown')}):")
        print(f"  Running: {status['running']}")
        print(f"  ADB Connected: {status['adb_connected']}")
    
    # Note: BlueStacks instances must be created manually
    print("\n=== Note: BlueStacks instances must be created manually ===")
    print("Please create and start instances through the BlueStacks application.")
    
    # Use first available instance
    if instances:
        vm_index = instances[0].get('index')
        print(f"\nUsing instance {vm_index}")
    else:
        print("No instances available")
        return
    
    # Set as active VM
    controller.set_active_vm(vm_index)
    print(f"Active VM set to: {controller.get_active_vm()}")
    
    # Wait a bit for VM to fully boot
    print("\nWaiting for VM to be fully ready...")
    time.sleep(5)
    
    # Connect ADB if not already connected
    if not controller.get_vm_status(vm_index)['adb_connected']:
        print("Connecting via ADB...")
        controller.connect_adb(vm_index)
    
    # Perform various ADB operations
    print("\n=== ADB Operations ===")
    
    # Get system information
    print("\nSystem Information:")
    info_commands = {
        "Android Version": "getprop ro.build.version.release",
        "SDK Version": "getprop ro.build.version.sdk",
        "Device Model": "getprop ro.product.model",
        "Manufacturer": "getprop ro.product.manufacturer",
        "Screen Resolution": "wm size",
    }
    
    for label, command in info_commands.items():
        result = controller.execute_adb_command(vm_index, command)
        print(f"  {label}: {result}")
    
    # Get battery information
    print("\nBattery Information:")
    battery_level = controller.execute_adb_command(vm_index, "dumpsys battery | grep level")
    print(f"  {battery_level}")
    
    # List top processes
    print("\nTop 5 Processes by CPU:")
    top_processes = controller.execute_adb_command(vm_index, "top -n 1 | head -6")
    print(top_processes)
    
    # Example: Install APK (uncomment and provide path to test)
    # print("\n=== Installing APK ===")
    # apk_path = "path/to/your/app.apk"
    # if controller.install_apk(vm_index, apk_path):
    #     print(f"APK installed successfully: {apk_path}")
    # else:
    #     print(f"Failed to install APK: {apk_path}")
    
    # Example: Launch an app (replace with actual package name)
    # print("\n=== Launching App ===")
    # package_name = "com.android.settings"
    # result = controller.execute_adb_command(
    #     vm_index, 
    #     f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
    # )
    # print(f"Launch result: {result}")
    
    # Cleanup: Disconnect ADB
    print("\n=== Cleanup ===")
    response = input("Do you want to disconnect ADB? (y/n): ")
    if response.lower() == 'y':
        controller.disconnect_adb(vm_index)
        print("ADB disconnected")
    else:
        print("ADB left connected")


if __name__ == "__main__":
    main()

