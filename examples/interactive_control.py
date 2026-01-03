"""
Interactive control example for MEmu controller.

This example provides an interactive command-line interface for controlling VMs.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing memu_controller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from memu_controller import MemuController, MemuConfig


def print_menu():
    """Print the main menu."""
    print("\n" + "="*50)
    print("MEmu Controller - Interactive Mode")
    print("="*50)
    print("1. List all VMs")
    print("2. Create new VM")
    print("3. Start VM")
    print("4. Stop VM")
    print("5. Delete VM")
    print("6. Get VM status")
    print("7. Connect ADB")
    print("8. Execute ADB command")
    print("9. Install APK")
    print("10. Set active VM")
    print("0. Exit")
    print("="*50)


def list_vms(controller: MemuController):
    """List all VMs."""
    vms = controller.list_vms()
    if not vms:
        print("No VMs found")
        return
    
    print(f"\nFound {len(vms)} VM(s):")
    for vm in vms:
        vm_index = vm.get('index')
        status = controller.get_vm_status(vm_index)
        print(f"\n  VM {vm_index}: {vm.get('name', 'Unknown')}")
        print(f"    Running: {status['running']}")
        print(f"    ADB Connected: {status['adb_connected']}")


def create_vm(controller: MemuController):
    """Create a new VM."""
    name = input("Enter VM name (or press Enter for default): ").strip()
    vm_name = name if name else None
    
    print("Creating VM...")
    vm_index = controller.create_and_start_vm(vm_name=vm_name)
    
    if vm_index is not None:
        print(f"VM created and started with index: {vm_index}")
    else:
        print("Failed to create VM")


def start_vm(controller: MemuController):
    """Start a VM."""
    try:
        vm_index = int(input("Enter VM index: "))
        print(f"Starting VM {vm_index}...")
        if controller.start_vm(vm_index):
            print("VM started successfully")
        else:
            print("Failed to start VM")
    except ValueError:
        print("Invalid VM index")


def stop_vm(controller: MemuController):
    """Stop a VM."""
    try:
        vm_index = int(input("Enter VM index: "))
        print(f"Stopping VM {vm_index}...")
        if controller.stop_vm(vm_index):
            print("VM stopped successfully")
        else:
            print("Failed to stop VM")
    except ValueError:
        print("Invalid VM index")


def delete_vm(controller: MemuController):
    """Delete a VM."""
    try:
        vm_index = int(input("Enter VM index: "))
        confirm = input(f"Are you sure you want to delete VM {vm_index}? (yes/no): ")
        if confirm.lower() == 'yes':
            print(f"Deleting VM {vm_index}...")
            if controller.delete_vm(vm_index):
                print("VM deleted successfully")
            else:
                print("Failed to delete VM")
        else:
            print("Deletion cancelled")
    except ValueError:
        print("Invalid VM index")


def get_vm_status(controller: MemuController):
    """Get VM status."""
    try:
        vm_index = int(input("Enter VM index: "))
        status = controller.get_vm_status(vm_index)
        print(f"\nVM {vm_index} Status:")
        print(f"  Running: {status['running']}")
        print(f"  ADB Connected: {status['adb_connected']}")
        if status['info']:
            print(f"  Name: {status['info'].get('name', 'Unknown')}")
    except ValueError:
        print("Invalid VM index")


def connect_adb(controller: MemuController):
    """Connect to VM via ADB."""
    try:
        vm_index = int(input("Enter VM index: "))
        print(f"Connecting to VM {vm_index} via ADB...")
        if controller.connect_adb(vm_index):
            print("ADB connected successfully")
        else:
            print("Failed to connect ADB")
    except ValueError:
        print("Invalid VM index")


def execute_adb_command(controller: MemuController):
    """Execute an ADB command."""
    try:
        vm_index = int(input("Enter VM index: "))
        command = input("Enter ADB command: ").strip()
        print(f"Executing: {command}")
        result = controller.execute_adb_command(vm_index, command)
        if result:
            print(f"Result:\n{result}")
        else:
            print("Command failed or returned no output")
    except ValueError:
        print("Invalid VM index")


def install_apk(controller: MemuController):
    """Install an APK."""
    try:
        vm_index = int(input("Enter VM index: "))
        apk_path = input("Enter path to APK file: ").strip()
        print(f"Installing APK: {apk_path}")
        if controller.install_apk(vm_index, apk_path):
            print("APK installed successfully")
        else:
            print("Failed to install APK")
    except ValueError:
        print("Invalid VM index")


def set_active_vm(controller: MemuController):
    """Set active VM."""
    try:
        vm_index = int(input("Enter VM index: "))
        if controller.set_active_vm(vm_index):
            print(f"Active VM set to: {vm_index}")
        else:
            print("Failed to set active VM (VM may not exist)")
    except ValueError:
        print("Invalid VM index")


def main():
    """Main interactive loop."""
    print("Initializing MEmu Controller...")
    controller = MemuController()
    
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

