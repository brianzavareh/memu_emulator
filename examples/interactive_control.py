"""
Interactive control example for BlueStacks controller.

This example provides an interactive command-line interface for controlling BlueStacks instances.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing android_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from android_controller import BlueStacksController, BlueStacksConfig


def print_menu():
    """Print the main menu."""
    print("\n" + "="*50)
    print("BlueStacks Controller - Interactive Mode")
    print("="*50)
    print("1. List all instances")
    print("2. Note: Instances must be created manually in BlueStacks")
    print("3. Check instance status and connect ADB")
    print("4. Disconnect ADB")
    print("5. Note: Instances must be deleted manually in BlueStacks")
    print("6. Get instance status")
    print("7. Connect ADB")
    print("8. Execute ADB command")
    print("9. Install APK")
    print("10. Set active instance")
    print("0. Exit")
    print("="*50)


def list_vms(controller: BlueStacksController):
    """List all BlueStacks instances."""
    instances = controller.list_vms()
    if not instances:
        print("No BlueStacks instances found")
        return
    
    print(f"\nFound {len(instances)} instance(s):")
    for instance in instances:
        vm_index = instance.get('index')
        status = controller.get_vm_status(vm_index)
        print(f"\n  Instance {vm_index}: {instance.get('name', 'Unknown')}")
        print(f"    Running: {status['running']}")
        print(f"    ADB Connected: {status['adb_connected']}")


def create_vm(controller: BlueStacksController):
    """Note about creating instances."""
    print("Note: BlueStacks instances must be created manually through the BlueStacks application.")


def start_vm(controller: BlueStacksController):
    """Check instance status and connect ADB."""
    try:
        vm_index = int(input("Enter instance index: "))
        print(f"Checking instance {vm_index} status...")
        status = controller.get_vm_status(vm_index)
        if status['running']:
            if not status['adb_connected']:
                print("Connecting ADB...")
                if controller.start_vm(vm_index, connect_adb=True):
                    print("Instance is running and ADB connected successfully")
                else:
                    print("Failed to connect ADB")
            else:
                print("Instance is running and ADB is already connected")
        else:
            print("Instance is not running. Please start it manually in BlueStacks.")
    except ValueError:
        print("Invalid instance index")


def stop_vm(controller: BlueStacksController):
    """Disconnect ADB."""
    try:
        vm_index = int(input("Enter instance index: "))
        print(f"Disconnecting ADB from instance {vm_index}...")
        if controller.stop_vm(vm_index, disconnect_adb=True):
            print("ADB disconnected successfully")
        else:
            print("Failed to disconnect ADB")
    except ValueError:
        print("Invalid instance index")


def delete_vm(controller: BlueStacksController):
    """Note about deleting instances."""
    print("Note: BlueStacks instances must be deleted manually through the BlueStacks application.")


def get_vm_status(controller: BlueStacksController):
    """Get instance status."""
    try:
        vm_index = int(input("Enter instance index: "))
        status = controller.get_vm_status(vm_index)
        print(f"\nInstance {vm_index} Status:")
        print(f"  Running: {status['running']}")
        print(f"  ADB Connected: {status['adb_connected']}")
        if status['info']:
            print(f"  Name: {status['info'].get('name', 'Unknown')}")
    except ValueError:
        print("Invalid instance index")


def connect_adb(controller: BlueStacksController):
    """Connect to instance via ADB."""
    try:
        vm_index = int(input("Enter instance index: "))
        print(f"Connecting to instance {vm_index} via ADB...")
        if controller.connect_adb(vm_index):
            print("ADB connected successfully")
        else:
            print("Failed to connect ADB")
    except ValueError:
        print("Invalid instance index")


def execute_adb_command(controller: BlueStacksController):
    """Execute an ADB command."""
    try:
        vm_index = int(input("Enter instance index: "))
        command = input("Enter ADB command: ").strip()
        print(f"Executing: {command}")
        result = controller.execute_adb_command(vm_index, command)
        if result:
            print(f"Result:\n{result}")
        else:
            print("Command failed or returned no output")
    except ValueError:
        print("Invalid instance index")


def install_apk(controller: BlueStacksController):
    """Install an APK."""
    try:
        vm_index = int(input("Enter instance index: "))
        apk_path = input("Enter path to APK file: ").strip()
        print(f"Installing APK: {apk_path}")
        if controller.install_apk(vm_index, apk_path):
            print("APK installed successfully")
        else:
            print("Failed to install APK")
    except ValueError:
        print("Invalid instance index")


def set_active_vm(controller: BlueStacksController):
    """Set active instance."""
    try:
        vm_index = int(input("Enter instance index: "))
        if controller.set_active_vm(vm_index):
            print(f"Active instance set to: {vm_index}")
        else:
            print("Failed to set active instance (instance may not exist)")
    except ValueError:
        print("Invalid instance index")


def main():
    """Main interactive loop."""
    print("Initializing BlueStacks Controller...")
    controller = BlueStacksController()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            list_vms(controller)
        elif choice == '2':
            create_vm(controller)
        elif choice == '3':
            start_vm(controller)
        elif choice == '4':
            stop_vm(controller)
        elif choice == '5':
            delete_vm(controller)
        elif choice == '6':
            get_vm_status(controller)
        elif choice == '7':
            connect_adb(controller)
        elif choice == '8':
            execute_adb_command(controller)
        elif choice == '9':
            install_apk(controller)
        elif choice == '10':
            set_active_vm(controller)
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

