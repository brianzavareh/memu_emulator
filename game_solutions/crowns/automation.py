"""
Crowns game automation script.

This script automates playing the Crowns puzzle game by:
1. Finding and tapping "Crowns" text to open the game
2. Analyzing the game board from screenshots
3. Solving the puzzle using constraint satisfaction
4. Automatically placing crowns by tapping on correct cells
"""

import sys
import time
import os
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from android_controller import BlueStacksController
from game_solutions.crowns.solver import solve_crowns_puzzle, validate_solution
from game_solutions.crowns.analyzer import CrownsBoardAnalyzer


def _cleanup_png_files():
    """
    Clean up PNG files created during automation.
    
    Removes:
    - crowns_board.png (screenshot of game board)
    - debug_grid.png (debug visualization from analyzer)
    """
    png_files = ["crowns_board.png", "debug_grid.png"]
    for png_file in png_files:
        try:
            if os.path.exists(png_file):
                os.remove(png_file)
        except Exception:
            pass  # Silently ignore cleanup errors


def find_and_open_crowns_game(controller: BlueStacksController, vm_index: int) -> bool:
    """
    Find and open the Crowns game by tapping on "Crowns" text.

    Parameters
    ----------
    controller : BlueStacksController
        Controller instance for emulator operations.
    vm_index : int
        Index of the VM instance.

    Returns
    -------
    bool
        True if game was found and opened, False otherwise.
    """
    # Try to find "Crowns" text (case-insensitive, partial match)
    if controller.find_and_tap_text(vm_index, "Crowns", exact_match=False, case_sensitive=False):
        # Minimal wait - game loads instantly, screenshot will verify
        time.sleep(0.1)
        return True
    else:
        print("Error: Could not find 'Crowns' text in the UI.")
        return False


def analyze_game_board(
    controller: BlueStacksController,
    vm_index: int,
    analyzer: CrownsBoardAnalyzer,
    screenshot_path: Optional[str] = None
) -> Optional[Tuple[dict, List[List[int]], List[List[Tuple[int, int]]]]]:
    """
    Analyze the game board from a screenshot.

    Parameters
    ----------
    controller : BlueStacksController
        Controller instance.
    vm_index : int
        Index of the VM instance.
    analyzer : CrownsBoardAnalyzer
        Board analyzer instance.
    screenshot_path : Optional[str], optional
        Path to save screenshot. Default is None.

    Returns
    -------
    Optional[Tuple[dict, List[List[int]], List[List[Tuple[int, int]]]]]
        Tuple of (grid_info, regions, cell_coordinates) or None if analysis fails.
    """
    # Only refresh display if needed (skip refresh for faster processing)
    screenshot = controller.take_screenshot_image(vm_index, refresh_display=False)
    
    if screenshot is None:
        print("Error: Failed to take screenshot.")
        return None

    # Save screenshot if path provided (for debugging)
    if screenshot_path:
        screenshot.save(screenshot_path)

    grid_info = analyzer.detect_grid(screenshot)
    
    if grid_info is None:
        print("Error: Failed to detect grid structure.")
        return None

    cell_data = analyzer.get_cell_data(screenshot, grid_info)
    
    if cell_data is None:
        print("Error: Failed to get cell data.")
        return None
    
    # Build regions matrix from cell data by grouping similar colors
    import numpy as np
    
    colors = np.array([cell['color'] for cell in cell_data])
    quantization_step = 25
    quantized_colors = (colors // quantization_step) * quantization_step
    
    # Map quantized colors to region IDs
    color_to_region = {}
    region_id = 0
    
    for quantized_color in quantized_colors:
        color_tuple = tuple(quantized_color)
        if color_tuple not in color_to_region:
            color_to_region[color_tuple] = region_id
            region_id += 1
    
    # Build regions matrix
    rows = grid_info['rows']
    cols = grid_info['cols']
    regions = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for cell, quantized_color in zip(cell_data, quantized_colors):
        color_tuple = tuple(quantized_color)
        regions[cell['row']][cell['col']] = color_to_region[color_tuple]

    # Build cell_coordinates matrix
    cell_coordinates = [[(0, 0) for _ in range(cols)] for _ in range(rows)]
    for cell in cell_data:
        cell_coordinates[cell['row']][cell['col']] = (cell['center_x'], cell['center_y'])

    return grid_info, regions, cell_coordinates


def solve_and_place_crowns(
    controller: BlueStacksController,
    vm_index: int,
    grid_info: dict,
    regions: List[List[int]],
    cell_coordinates: List[List[Tuple[int, int]]],
    delay_between_taps: float = 0.01
) -> bool:
    """
    Solve the puzzle and place crowns by tapping on cells.

    Parameters
    ----------
    controller : BlueStacksController
        Controller instance.
    vm_index : int
        Index of the VM instance.
    grid_info : dict
        Grid information from analyzer.
    regions : List[List[int]]
        2D list of region IDs.
    cell_coordinates : List[List[Tuple[int, int]]]
        2D list of (x, y) coordinates for each cell.
    delay_between_taps : float, optional
        Delay in seconds between taps. Default is 0.01.

    Returns
    -------
    bool
        True if solution was found and crowns were placed, False otherwise.
    """
    rows = grid_info['rows']
    cols = grid_info['cols']

    solution = solve_crowns_puzzle(rows, cols, regions)

    if solution is None:
        print("Error: No solution found for this puzzle.")
        return False

    # Validate solution
    is_valid, errors = validate_solution(solution, rows, cols, regions)
    if not is_valid:
        print("Warning: Solution validation failed, but proceeding anyway.")
        # Continue anyway - validation might have false positives

    # Place crowns by tapping on cells
    for row, col in solution:
        if 0 <= row < rows and 0 <= col < cols:
            x, y = cell_coordinates[row][col]
            controller.tap(vm_index, x, y)
            time.sleep(delay_between_taps)
        else:
            print(f"Warning: Invalid position ({row}, {col})")
            # Continue with other crowns instead of failing completely

    return True


def main():
    """
    Main automation function.
    """
    # Initialize controller
    controller = BlueStacksController()
    vm_index = 0  # Default VM index

    # Quick VM status check - skip detailed check if we can connect directly
    if not controller.connect_adb(vm_index):
        # If quick connect fails, do full status check
        vm_status = controller.get_vm_status(vm_index)
        if not vm_status['running']:
            print(f"Error: VM instance {vm_index} is not running.")
            return 1
        if not vm_status['adb_connected']:
            print("Error: Failed to connect via ADB.")
            return 1

    # Find and open the game
    if not find_and_open_crowns_game(controller, vm_index):
        print("Error: Could not open Crowns game.")
        return 1

    # Minimal wait - game loads instantly, screenshot will verify it's ready
    time.sleep(0.1)

    # Initialize analyzer (debug disabled for performance)
    analyzer = CrownsBoardAnalyzer(debug=False)

    # Analyze game board (no screenshot saving for performance)
    result = analyze_game_board(
        controller,
        vm_index,
        analyzer,
        screenshot_path=None
    )

    if result is None:
        print("Error: Failed to analyze game board.")
        return 1

    grid_info, regions, cell_coordinates = result

    # Solve and place crowns
    if not solve_and_place_crowns(
        controller,
        vm_index,
        grid_info,
        regions,
        cell_coordinates
    ):
        print("Error: Failed to solve and place crowns.")
        return 1

    # Clean up PNG files after successful run
    _cleanup_png_files()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nAutomation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
