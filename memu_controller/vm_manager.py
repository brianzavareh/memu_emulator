"""
Virtual Machine management module.

This module handles VM lifecycle operations: creation, starting, stopping,
listing, and deletion of MEmu virtual machines.
"""

from typing import List, Optional, Dict, Any
from pymemuc import PyMemuc


class VMManager:
    """
    Manager class for MEmu virtual machine operations.

    This class provides a high-level interface for managing MEmu VMs,
    including creation, deletion, starting, stopping, and querying VM status.

    Attributes
    ----------
    memuc : PyMemuc
        Instance of PyMemuc for interacting with MEmu.
    """

    def __init__(self, memuc_path: Optional[str] = None):
        """
        Initialize the VM Manager.

        Parameters
        ----------
        memuc_path : Optional[str], optional
            Path to memuc.exe. If None, pymemuc will attempt to auto-detect.
        """
        self.memuc = PyMemuc(memuc_path=memuc_path)

    def list_vms(self) -> List[Dict[str, Any]]:
        """
        List all available virtual machines.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing VM information.
            Each dictionary contains 'index' and 'name' keys.
        """
        try:
            vm_list = self.memuc.list_vm_info()
            return vm_list if vm_list else []
        except Exception as e:
            print(f"Error listing VMs: {e}")
            return []

    def get_vm_info(self, vm_index: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific VM.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing VM information, or None if not found.
        """
        vms = self.list_vms()
        for vm in vms:
            if vm.get("index") == vm_index:
                return vm
        return None

    def create_vm(self, vm_name: Optional[str] = None) -> Optional[int]:
        """
        Create a new virtual machine.

        Parameters
        ----------
        vm_name : Optional[str], optional
            Name for the new VM. If None, MEmu will assign a default name.

        Returns
        -------
        Optional[int]
            Index of the created VM, or None if creation failed.
        """
        try:
            if vm_name:
                vm_index = self.memuc.create_vm(vm_name=vm_name)
            else:
                vm_index = self.memuc.create_vm()
            return vm_index
        except Exception as e:
            print(f"Error creating VM: {e}")
            return None

    def start_vm(self, vm_index: int) -> bool:
        """
        Start a virtual machine.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to start.

        Returns
        -------
        bool
            True if VM was started successfully, False otherwise.
        """
        try:
            result = self.memuc.start_vm(vm_index)
            return result is not False
        except Exception as e:
            print(f"Error starting VM {vm_index}: {e}")
            return False

    def stop_vm(self, vm_index: int) -> bool:
        """
        Stop a running virtual machine.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to stop.

        Returns
        -------
        bool
            True if VM was stopped successfully, False otherwise.
        """
        try:
            result = self.memuc.stop_vm(vm_index)
            return result is not False
        except Exception as e:
            print(f"Error stopping VM {vm_index}: {e}")
            return False

    def delete_vm(self, vm_index: int) -> bool:
        """
        Delete a virtual machine.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to delete.

        Returns
        -------
        bool
            True if VM was deleted successfully, False otherwise.
        """
        try:
            result = self.memuc.delete_vm(vm_index)
            return result is not False
        except Exception as e:
            print(f"Error deleting VM {vm_index}: {e}")
            return False

    def is_vm_running(self, vm_index: int) -> bool:
        """
        Check if a virtual machine is currently running.

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine to check.

        Returns
        -------
        bool
            True if VM is running, False otherwise.
        """
        try:
            running_vms = self.memuc.list_running()
            return vm_index in running_vms if running_vms else False
        except Exception as e:
            print(f"Error checking VM status {vm_index}: {e}")
            return False

    def wait_for_vm_ready(self, vm_index: int, timeout: int = 60) -> bool:
        """
        Wait for a VM to be ready (booted and responsive).

        Parameters
        ----------
        vm_index : int
            Index of the virtual machine.
        timeout : int, optional
            Maximum time to wait in seconds. Default is 60.

        Returns
        -------
        bool
            True if VM is ready within timeout, False otherwise.
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_vm_running(vm_index):
                # Additional check: try to get VM info to ensure it's responsive
                if self.get_vm_info(vm_index) is not None:
                    return True
            time.sleep(2)
        
        return False

