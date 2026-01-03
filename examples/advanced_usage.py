"""
Advanced usage example for MEmu controller.

This example demonstrates advanced operations including custom configuration,
multiple VM management, and APK installation.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing memu_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import MemuController, MemuConfig
import time


def main():
    """Demonstrate advanced MEmu controller usage."""
    # Create custom configuration
    config = MemuConfig(
        default_vm_name="Advanced_Test_VM",
        auto_start=True,
        timeout=90,
        adb_port_base=21503
    )
    
    controller = MemuController(config=config)
    
    # List all VMs and their status
    print("=== VM Status Overview ===")
    vms = controller.list_vms()
    for vm in vms:
        vm_index = vm.get('index')
        status = controller.get_vm_status(vm_index)
        print(f"\nVM {vm_index} ({vm.get('name', 'Unknown')}):")
        print(f"  Running: {status['running']}")
        print(f"  ADB Connected: {status['adb_connected']}")
    
    # Create a new VM with custom name
    print("\n=== Creating New VM ===")
    vm_index = controller.create_and_start_vm(vm_name="My_Custom_VM")
    
    if vm_index is None:
        print("Failed to create VM")
        return
    
    print(f"VM created with index: {vm_index}")
    
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
    
    # Cleanup: Stop and optionally delete VM
    print("\n=== Cleanup ===")
    response = input("Do you want to stop the VM? (y/n): ")
    if response.lower() == 'y':
        controller.stop_vm(vm_index)
        print("VM stopped")
        
        response = input("Do you want to delete the VM? (y/n): ")
        if response.lower() == 'y':
            controller.delete_vm(vm_index)
            print("VM deleted")
    else:
        print("VM left running")


if __name__ == "__main__":
    main()

