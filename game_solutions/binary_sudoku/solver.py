"""
Binary Sudoku+ puzzle solver using constraint satisfaction.

This module implements a backtracking algorithm to solve Binary Sudoku+ puzzles
with the following constraints:
- Each row must contain equal number of 0s and 1s
- Each column must contain equal number of 0s and 1s
- No more than two consecutive same numbers in rows or columns
- No two rows are the same
- No two columns are the same
- Cells connected by '=' must have same value
- Cells connected by '≠' must have different values
"""

from typing import List, Tuple, Optional, Set, Dict
import copy


def count_row_values(grid: List[List[Optional[int]]], row: int, value: int) -> int:
    """
    Count occurrences of a value in a row.

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state (None = empty, 0 = zero, 1 = one).
    row : int
        Row index (0-based).
    value : int
        Value to count (0 or 1).

    Returns
    -------
    int
        Number of occurrences of value in the row.
    """
    return sum(1 for cell in grid[row] if cell == value)


def count_col_values(grid: List[List[Optional[int]]], col: int, value: int) -> int:
    """
    Count occurrences of a value in a column.

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state.
    col : int
        Column index (0-based).
    value : int
        Value to count (0 or 1).

    Returns
    -------
    int
        Number of occurrences of value in the column.
    """
    return sum(1 for r in range(len(grid)) if grid[r][col] == value)


def has_too_many_consecutive(
    grid: List[List[Optional[int]]],
    row: int,
    col: int,
    value: int,
    rows: int,
    cols: int
) -> bool:
    """
    Check if placing value at (row, col) creates more than 2 consecutive same values.

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state.
    row : int
        Row index.
    col : int
        Column index.
    value : int
        Value to place (0 or 1).
    rows : int
        Total number of rows.
    cols : int
        Total number of columns.

    Returns
    -------
    bool
        True if placing value would create more than 2 consecutive same values.
    """
    # Check horizontal (row)
    consecutive_count = 1  # Count the cell we're placing
    
    # Check left
    for c in range(col - 1, -1, -1):
        if grid[row][c] == value:
            consecutive_count += 1
        else:
            break
    
    # Check right
    for c in range(col + 1, cols):
        if grid[row][c] == value:
            consecutive_count += 1
        else:
            break
    
    if consecutive_count > 2:
        return True
    
    # Check vertical (column)
    consecutive_count = 1
    
    # Check up
    for r in range(row - 1, -1, -1):
        if grid[r][col] == value:
            consecutive_count += 1
        else:
            break
    
    # Check down
    for r in range(row + 1, rows):
        if grid[r][col] == value:
            consecutive_count += 1
        else:
            break
    
    if consecutive_count > 2:
        return True
    
    return False


def rows_are_duplicate(
    grid: List[List[Optional[int]]],
    row1: int,
    row2: int
) -> bool:
    """
    Check if two rows are identical (ignoring empty cells).

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state.
    row1 : int
        First row index.
    row2 : int
        Second row index.

    Returns
    -------
    bool
        True if rows are identical (when both are complete).
    """
    # Only check if both rows are complete (no None values)
    if any(cell is None for cell in grid[row1]) or any(cell is None for cell in grid[row2]):
        return False
    
    return grid[row1] == grid[row2]


def cols_are_duplicate(
    grid: List[List[Optional[int]]],
    col1: int,
    col2: int
) -> bool:
    """
    Check if two columns are identical (ignoring empty cells).

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state.
    col1 : int
        First column index.
    col2 : int
        Second column index.

    Returns
    -------
    bool
        True if columns are identical (when both are complete).
    """
    rows = len(grid)
    
    # Only check if both columns are complete
    if any(grid[r][col1] is None for r in range(rows)) or any(grid[r][col2] is None for r in range(rows)):
        return False
    
    return all(grid[r][col1] == grid[r][col2] for r in range(rows))


def is_valid_placement(
    grid: List[List[Optional[int]]],
    row: int,
    col: int,
    value: int,
    rows: int,
    cols: int,
    constraints: Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]] = None
) -> bool:
    """
    Check if placing value at (row, col) violates any constraints.

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Current grid state.
    row : int
        Row index.
    col : int
        Column index.
    value : int
        Value to place (0 or 1).
    rows : int
        Total number of rows.
    cols : int
        Total number of columns.
    constraints : Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]], optional
        Dictionary mapping cell pairs to constraint type ('=' or '≠').
        Default is None.

    Returns
    -------
    bool
        True if placement is valid, False otherwise.
    """
    # Check if cell is already filled
    if grid[row][col] is not None:
        return False
    
    # Check constraint markers first (before creating temp grid)
    if constraints:
        for (cell1, cell2), constraint in constraints.items():
            r1, c1 = cell1
            r2, c2 = cell2
            
            if (row, col) == (r1, c1) or (row, col) == (r2, c2):
                other_row, other_col = (r2, c2) if (row, col) == (r1, c1) else (r1, c1)
                other_val = grid[other_row][other_col]
                
                if other_val is not None:
                    if constraint == '=' and value != other_val:
                        return False
                    if constraint == '≠' and value == other_val:
                        return False
    
    # Create temporary grid with the placement
    temp_grid = copy.deepcopy(grid)
    temp_grid[row][col] = value
    
    # Check row constraint: equal 0s and 1s
    row_zeros = count_row_values(temp_grid, row, 0)
    row_ones = count_row_values(temp_grid, row, 1)
    row_empty = sum(1 for cell in temp_grid[row] if cell is None)
    
    # If row is complete, must have equal 0s and 1s
    if row_empty == 0:
        if row_zeros != row_ones:
            return False
    else:
        # If row incomplete, check if it's still possible
        if abs(row_zeros - row_ones) > row_empty:
            return False
    
    # Check column constraint: equal 0s and 1s
    col_zeros = count_col_values(temp_grid, col, 0)
    col_ones = count_col_values(temp_grid, col, 1)
    col_empty = sum(1 for r in range(rows) if temp_grid[r][col] is None)
    
    if col_empty == 0:
        if col_zeros != col_ones:
            return False
    else:
        if abs(col_zeros - col_ones) > col_empty:
            return False
    
    # Check consecutive constraint
    if has_too_many_consecutive(temp_grid, row, col, value, rows, cols):
        return False
    
    # Check duplicate rows (only if row is complete)
    if row_empty == 0:
        for r in range(rows):
            if r != row and rows_are_duplicate(temp_grid, row, r):
                return False
    
    # Check duplicate columns (only if column is complete)
    if col_empty == 0:
        for c in range(cols):
            if c != col and cols_are_duplicate(temp_grid, col, c):
                return False
    
    return True


def solve_binary_sudoku_puzzle(
    grid: List[List[Optional[int]]],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    constraints: Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]] = None
) -> Optional[List[List[int]]]:
    """
    Solve a Binary Sudoku+ puzzle using backtracking.

    Parameters
    ----------
    grid : List[List[Optional[int]]]
        Initial grid state. None = empty, 0 = zero, 1 = one.
    rows : Optional[int], optional
        Number of rows. If None, inferred from grid. Default is None.
    cols : Optional[int], optional
        Number of columns. If None, inferred from grid. Default is None.
    constraints : Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]], optional
        Dictionary mapping cell pairs to constraint type ('=' or '≠').
        Default is None.

    Returns
    -------
    Optional[List[List[int]]]
        Solved grid as 2D list of 0s and 1s, or None if no solution exists.
    """
    if not grid or not grid[0]:
        return None
    
    if rows is None:
        rows = len(grid)
    if cols is None:
        cols = len(grid[0])
    
    # Create working copy
    working_grid = copy.deepcopy(grid)
    
    # Find all empty cells
    empty_cells = []
    for r in range(rows):
        for c in range(cols):
            if working_grid[r][c] is None:
                empty_cells.append((r, c))
    
    def backtrack(cell_idx: int) -> bool:
        """
        Recursive backtracking function.

        Parameters
        ----------
        cell_idx : int
            Index of current cell being processed.

        Returns
        -------
        bool
            True if solution found, False otherwise.
        """
        # Base case: all cells filled
        if cell_idx >= len(empty_cells):
            # Verify all constraints are satisfied
            for r in range(rows):
                if count_row_values(working_grid, r, 0) != count_row_values(working_grid, r, 1):
                    return False
            for c in range(cols):
                if count_col_values(working_grid, c, 0) != count_col_values(working_grid, c, 1):
                    return False
            
            # Check no duplicate rows
            for r1 in range(rows):
                for r2 in range(r1 + 1, rows):
                    if rows_are_duplicate(working_grid, r1, r2):
                        return False
            
            # Check no duplicate columns
            for c1 in range(cols):
                for c2 in range(c1 + 1, cols):
                    if cols_are_duplicate(working_grid, c1, c2):
                        return False
            
            return True
        
        row, col = empty_cells[cell_idx]
        
        # Try placing 0 and 1
        for value in [0, 1]:
            if is_valid_placement(working_grid, row, col, value, rows, cols, constraints):
                working_grid[row][col] = value
                
                if backtrack(cell_idx + 1):
                    return True
                
                # Backtrack
                working_grid[row][col] = None
        
        return False
    
    # Start backtracking
    if backtrack(0):
        # Convert to list of lists of ints (no None values)
        solution = []
        for r in range(rows):
            solution.append([working_grid[r][c] for c in range(cols)])
        return solution
    
    return None


def validate_solution(
    solution: List[List[int]],
    rows: Optional[int] = None,
    cols: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that a solution satisfies all Binary Sudoku+ constraints.

    Parameters
    ----------
    solution : List[List[int]]
        Solution grid (2D list of 0s and 1s).
    rows : Optional[int], optional
        Number of rows. If None, inferred from solution. Default is None.
    cols : Optional[int], optional
        Number of columns. If None, inferred from solution. Default is None.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors) tuple.
    """
    errors = []
    
    if not solution or not solution[0]:
        errors.append("Solution is empty")
        return False, errors
    
    if rows is None:
        rows = len(solution)
    if cols is None:
        cols = len(solution[0])
    
    # Check equal 0s and 1s in each row
    for r in range(rows):
        zeros = count_row_values(solution, r, 0)
        ones = count_row_values(solution, r, 1)
        if zeros != ones:
            errors.append(f"Row {r} has {zeros} zeros and {ones} ones (must be equal)")
    
    # Check equal 0s and 1s in each column
    for c in range(cols):
        zeros = count_col_values(solution, c, 0)
        ones = count_col_values(solution, c, 1)
        if zeros != ones:
            errors.append(f"Column {c} has {zeros} zeros and {ones} ones (must be equal)")
    
    # Check no more than 2 consecutive same values
    for r in range(rows):
        consecutive = 1
        for c in range(1, cols):
            if solution[r][c] == solution[r][c-1]:
                consecutive += 1
                if consecutive > 2:
                    errors.append(f"Row {r} has more than 2 consecutive {solution[r][c]}s")
                    break
            else:
                consecutive = 1
    
    for c in range(cols):
        consecutive = 1
        for r in range(1, rows):
            if solution[r][c] == solution[r-1][c]:
                consecutive += 1
                if consecutive > 2:
                    errors.append(f"Column {c} has more than 2 consecutive {solution[r][c]}s")
                    break
            else:
                consecutive = 1
    
    # Check no duplicate rows
    for r1 in range(rows):
        for r2 in range(r1 + 1, rows):
            if solution[r1] == solution[r2]:
                errors.append(f"Rows {r1} and {r2} are identical")
    
    # Check no duplicate columns
    for c1 in range(cols):
        for c2 in range(c1 + 1, cols):
            if all(solution[r][c1] == solution[r][c2] for r in range(rows)):
                errors.append(f"Columns {c1} and {c2} are identical")
    
    return len(errors) == 0, errors

