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
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from android_controller import BlueStacksController
from game_solutions.crowns.solver import solve_crowns_puzzle, validate_solution
from game_solutions.crowns.analyzer import CrownsBoardAnalyzer


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
    print("Searching for 'Crowns' game...")
    
    # Try to find "Crowns" text (case-insensitive, partial match)
    if controller.find_and_tap_text(vm_index, "Crowns", exact_match=False, case_sensitive=False):
        print("Successfully tapped on 'Crowns'!")
        time.sleep(2)  # Wait for game to load
        return True
    else:
        print("Could not find 'Crowns' text in the UI.")
        # Try to find coordinates for debugging
        coords = controller.find_text_coordinates(vm_index, "Crowns")
        if coords:
            print(f"Found 'Crowns' at center: {coords['center']}")
            print(f"Bounds: {coords['bounds']}")
            print("Attempting manual tap...")
            center_x, center_y = coords['center']
            controller.tap(vm_index, center_x, center_y)
            time.sleep(2)
            return True
        else:
            print("Text 'Crowns' not found in UI hierarchy.")
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
    print("\nTaking screenshot of game board...")
    screenshot = controller.take_screenshot_image(vm_index, refresh_display=True)
    
    if screenshot is None:
        print("Error: Failed to take screenshot.")
        return None

    # Save screenshot if path provided
    if screenshot_path:
        screenshot.save(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

    print("Analyzing grid structure...")
    grid_info = analyzer.detect_grid(screenshot)
    
    if grid_info is None:
        print("Error: Failed to detect grid structure.")
        return None

    print(f"Detected grid: {grid_info['rows']}x{grid_info['cols']}")
    print(f"Cell size: {grid_info['cell_width']}x{grid_info['cell_height']}")

    print("Getting cell data (coordinates and colors)...")
    cell_data = analyzer.get_cell_data(screenshot, grid_info)
    
    if cell_data is None:
        print("Error: Failed to get cell data.")
        return None

    print(f"Got data for {len(cell_data)} cells")
    print("\nSample cell data (first 5 cells):")
    for i, cell in enumerate(cell_data[:5]):
        print(f"  Cell ({cell['row']}, {cell['col']}): center=({cell['center_x']}, {cell['center_y']}), color={cell['color']}")
    
    # Build regions matrix from cell data by clustering colors
    # Group cells with similar colors into regions
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Extract colors
    colors = np.array([cell['color'] for cell in cell_data])
    
    # Cluster into 9 regions
    print("\nClustering colors into 9 regions...")
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    region_labels = kmeans.fit_predict(colors)
    
    # Build regions matrix
    rows = grid_info['rows']
    cols = grid_info['cols']
    regions = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for cell, label in zip(cell_data, region_labels):
        regions[cell['row']][cell['col']] = int(label)
    
    # Count unique regions
    unique_regions = set()
    for row in regions:
        unique_regions.update(row)
    print(f"Detected {len(unique_regions)} unique regions")

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
    delay_between_taps: float = 0.5
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
        Delay in seconds between taps. Default is 0.5.

    Returns
    -------
    bool
        True if solution was found and crowns were placed, False otherwise.
    """
    rows = grid_info['rows']
    cols = grid_info['cols']

    print(f"\nSolving puzzle ({rows}x{cols})...")
    solution = solve_crowns_puzzle(rows, cols, regions)

    if solution is None:
        print("Error: No solution found for this puzzle.")
        return False

    print(f"Solution found! Placing {len(solution)} crowns...")

    # Validate solution
    is_valid, errors = validate_solution(solution, rows, cols, regions)
    if not is_valid:
        print("Warning: Solution validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("Proceeding anyway...")

    # Place crowns by tapping on cells
    for i, (row, col) in enumerate(solution, 1):
        if 0 <= row < rows and 0 <= col < cols:
            x, y = cell_coordinates[row][col]
            print(f"Placing crown {i}/{len(solution)} at ({row}, {col}) -> ({x}, {y})")
            controller.tap(vm_index, x, y)
            time.sleep(delay_between_taps)
        else:
            print(f"Warning: Invalid position ({row}, {col})")

    print("\nAll crowns placed!")
    return True


def main():
    """
    Main automation function.
    """
    print("=" * 80)
    print("Crowns Game Automation")
    print("=" * 80)

    # Initialize controller
    controller = BlueStacksController()
    vm_index = 0  # Default VM index

    # Check VM status
    print("\nChecking VM status...")
    vm_status = controller.get_vm_status(vm_index)

    if not vm_status['running']:
        print(f"Error: VM instance {vm_index} is not running.")
        print("Please start the emulator manually.")
        return 1

    if not vm_status['adb_connected']:
        print("Connecting to VM via ADB...")
        if not controller.connect_adb(vm_index):
            print("Error: Failed to connect via ADB.")
            return 1
        print("ADB connected successfully.")
        time.sleep(1)

    # Find and open the game
    if not find_and_open_crowns_game(controller, vm_index):
        print("Error: Could not open Crowns game.")
        return 1

    # Wait a bit more for game to fully load
    print("Waiting for game to load...")
    time.sleep(3)

    # Initialize analyzer
    analyzer = CrownsBoardAnalyzer(debug=True)

    # Analyze game board
    result = analyze_game_board(
        controller,
        vm_index,
        analyzer,
        screenshot_path="crowns_board.png"
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

    print("\n" + "=" * 80)
    print("Automation completed successfully!")
    print("=" * 80)

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

