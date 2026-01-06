"""
Binary Sudoku+ game automation script.

This script automates playing the Binary Sudoku+ puzzle game by:
1. Finding and opening the Binary Sudoku+ game
2. Analyzing the game board from screenshots
3. Solving the puzzle using constraint satisfaction
4. Automatically filling cells by toggling to reach target values
"""

import sys
import time
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from android_controller import BlueStacksController
from game_solutions.binary_sudoku.solver import solve_binary_sudoku_puzzle, validate_solution
from game_solutions.binary_sudoku.analyzer import BinarySudokuBoardAnalyzer


def _cleanup_png_files():
    """
    Clean up PNG files created during automation.
    
    Removes:
    - binary_sudoku_board.png (screenshot of game board)
    - debug_binary_sudoku_grid.png (debug visualization from analyzer)
    """
    png_files = ["binary_sudoku_board.png", "debug_binary_sudoku_grid.png"]
    for png_file in png_files:
        try:
            if os.path.exists(png_file):
                os.remove(png_file)
        except Exception:
            pass  # Silently ignore cleanup errors


def find_and_open_binary_sudoku_game(controller: BlueStacksController, vm_index: int) -> bool:
    """
    Find and open the Binary Sudoku+ game by tapping on "How to Play Binary Sudoku+" text.

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
    # Try to find "How to Play Binary Sudoku+" text (from content-desc)
    # Use partial match since the full text might vary
    search_texts = [
        "How to Play Binary Sudoku+",
        "Binary Sudoku+",
        "Binary Sudoku"
    ]
    
    for search_text in search_texts:
        if controller.find_and_tap_text(vm_index, search_text, exact_match=False, case_sensitive=False):
            # Wait for game to load
            time.sleep(0.5)
            return True
    
    print("Error: Could not find Binary Sudoku+ game in the UI.")
    return False


def analyze_game_board(
    controller: BlueStacksController,
    vm_index: int,
    analyzer: BinarySudokuBoardAnalyzer,
    screenshot_path: Optional[str] = None
) -> Optional[Tuple[dict, List[List[Optional[int]]], List[List[Tuple[int, int]]], Dict]]:
    """
    Analyze the game board from a screenshot.

    Parameters
    ----------
    controller : BlueStacksController
        Controller instance.
    vm_index : int
        Index of the VM instance.
    analyzer : BinarySudokuBoardAnalyzer
        Board analyzer instance.
    screenshot_path : Optional[str], optional
        Path to save screenshot. Default is None.

    Returns
    -------
    Optional[Tuple[dict, List[List[Optional[int]]], List[List[Tuple[int, int]]], Dict]]
        Tuple of (grid_info, cell_values, cell_coordinates, constraints) or None if analysis fails.
    """
    # Take screenshot
    screenshot = controller.take_screenshot_image(vm_index, refresh_display=False)
    
    if screenshot is None:
        print("Error: Failed to take screenshot.")
        return None

    # Save screenshot if path provided (for debugging)
    if screenshot_path:
        screenshot.save(screenshot_path)

    # Detect grid structure
    grid_info = analyzer.detect_grid(screenshot)
    
    if grid_info is None:
        print("Error: Failed to detect grid structure.")
        return None

    # Extract cell values (0, 1, or None for empty)
    cell_values = analyzer.extract_cell_values(screenshot, grid_info)
    
    if cell_values is None:
        print("Error: Failed to extract cell values.")
        return None
    
    # Detect constraints (= and ≠ markers)
    constraints = analyzer.detect_constraints(screenshot, grid_info)
    
    # Get cell coordinates
    cell_coordinates = analyzer.get_cell_coordinates(grid_info)

    return grid_info, cell_values, cell_coordinates, constraints


def calculate_toggle_sequence(
    current_value: Optional[int],
    target_value: int
) -> int:
    """
    Calculate how many toggles are needed to reach target value.
    
    Toggle cycle: empty (None) → 0 → 1 → empty (None) → ...
    
    Parameters
    ----------
    current_value : Optional[int]
        Current cell value (None, 0, or 1).
    target_value : int
        Target cell value (0 or 1).

    Returns
    -------
    int
        Number of toggles needed (0, 1, or 2).
    """
    if current_value == target_value:
        return 0  # Already correct
    
    # Toggle cycle: None → 0 → 1 → None → ...
    if current_value is None:
        # Empty → need to toggle once to get 0, twice to get 1
        return 1 if target_value == 0 else 2
    elif current_value == 0:
        # 0 → need to toggle once to get 1, twice to get empty then 0 again
        return 1 if target_value == 1 else 2
    else:  # current_value == 1
        # 1 → need to toggle once to get empty, twice to get 0
        return 1 if target_value == 0 else 2


def solve_and_fill_grid(
    controller: BlueStacksController,
    vm_index: int,
    grid_info: dict,
    cell_values: List[List[Optional[int]]],
    cell_coordinates: List[List[Tuple[int, int]]],
    constraints: Dict,
    delay_between_taps: float = 0.05
) -> bool:
    """
    Solve the puzzle and fill cells by toggling to reach target values.

    Parameters
    ----------
    controller : BlueStacksController
        Controller instance.
    vm_index : int
        Index of the VM instance.
    grid_info : dict
        Grid information from analyzer.
    cell_values : List[List[Optional[int]]]
        Current cell values (2D list: None, 0, or 1).
    cell_coordinates : List[List[Tuple[int, int]]]
        2D list of (x, y) coordinates for each cell.
    constraints : Dict
        Dictionary mapping cell pairs to constraint type ('=' or '≠').
    delay_between_taps : float, optional
        Delay in seconds between taps. Default is 0.05.

    Returns
    -------
    bool
        True if solution was found and cells were filled, False otherwise.
    """
    rows = grid_info['rows']
    cols = grid_info['cols']

    # Solve the puzzle
    solution = solve_binary_sudoku_puzzle(cell_values, rows, cols, constraints)

    if solution is None:
        print("Error: No solution found for this puzzle.")
        return False

    # Validate solution
    is_valid, errors = validate_solution(solution, rows, cols)
    if not is_valid:
        print("Warning: Solution validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("Proceeding anyway...")

    # Calculate toggle sequences for each cell
    toggle_actions = []
    for row in range(rows):
        for col in range(cols):
            current_value = cell_values[row][col]
            target_value = solution[row][col]
            toggles_needed = calculate_toggle_sequence(current_value, target_value)
            
            if toggles_needed > 0:
                x, y = cell_coordinates[row][col]
                toggle_actions.append((row, col, x, y, toggles_needed))

    if not toggle_actions:
        print("Puzzle is already solved!")
        return True

    print(f"Found solution. Need to toggle {len(toggle_actions)} cells.")

    # Execute toggles
    for row, col, x, y, toggles_needed in toggle_actions:
        for toggle in range(toggles_needed):
            controller.tap(vm_index, x, y)
            time.sleep(delay_between_taps)
        
        if delay_between_taps < 0.1:
            time.sleep(0.05)  # Small delay between different cells

    return True


def main():
    """
    Main automation function.
    
    Assumes the user is already in the Binary Sudoku+ game screen or can find it.
    Takes a screenshot, analyzes the board, solves the puzzle, and fills cells.
    """
    # Initialize controller
    controller = BlueStacksController()
    vm_index = 0  # Default VM index

    # Connect to VM
    if not controller.connect_adb(vm_index):
        vm_status = controller.get_vm_status(vm_index)
        if not vm_status['running']:
            print(f"Error: VM instance {vm_index} is not running.")
            return 1
        if not vm_status['adb_connected']:
            print("Error: Failed to connect via ADB.")
            return 1
        if not controller.connect_adb(vm_index):
            print("Error: Failed to establish ADB connection after status check.")
            return 1

    # Try to find and open the game (optional - user might already be in game)
    print("Looking for Binary Sudoku+ game...")
    if not find_and_open_binary_sudoku_game(controller, vm_index):
        print("Warning: Could not find game. Assuming already in game screen.")
        time.sleep(0.5)  # Brief wait

    # Initialize analyzer (debug enabled to save screenshots)
    analyzer = BinarySudokuBoardAnalyzer(debug=True)

    # Analyze game board
    result = analyze_game_board(
        controller,
        vm_index,
        analyzer,
        screenshot_path="binary_sudoku_board.png"  # Save screenshot for debugging
    )

    if result is None:
        print("Error: Failed to analyze game board.")
        return 1

    grid_info, cell_values, cell_coordinates, constraints = result

    # Print current state for debugging
    rows = grid_info['rows']
    cols = grid_info['cols']
    print(f"Detected {rows}x{cols} grid")
    print("Current board state:")
    
    # Create table with borders
    # Top border
    top_border = "+" + "+".join(["---"] * cols) + "+"
    print(f"  {top_border}")
    
    # Print each row
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            value = cell_values[row][col]
            if value is None:
                cell_str = " . "
            else:
                cell_str = f" {value} "
            row_cells.append(cell_str)
        row_str = "|" + "|".join(row_cells) + "|"
        print(f"  {row_str}")
        
        # Row separator (except after last row)
        if row < rows - 1:
            separator = "+" + "+".join(["---"] * cols) + "+"
            print(f"  {separator}")
    
    # Bottom border
    print(f"  {top_border}")
    
    # Print constraints if any
    if constraints:
        print(f"\nDetected {len(constraints)} constraints:")
        for (cell1, cell2), constraint in constraints.items():
            print(f"  {cell1} <-> {cell2}: {constraint}")

    # Solve and fill grid
    if not solve_and_fill_grid(
        controller,
        vm_index,
        grid_info,
        cell_values,
        cell_coordinates,
        constraints
    ):
        print("Error: Failed to solve and fill grid.")
        return 1

    # Clean up PNG files after successful run
    _cleanup_png_files()

    print("Binary Sudoku+ puzzle solved successfully!")
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

