"""
Crowns puzzle solver using constraint satisfaction.

This module implements a backtracking algorithm to solve Crowns puzzles
with the following constraints:
- Each row must contain exactly one crown
- Each column must contain exactly one crown
- Each region (same background color) must contain exactly one crown
- No two crowns can be adjacent (horizontally, vertically, or diagonally)
"""

from typing import List, Tuple, Optional, Dict, Set
import copy


def get_adjacent_cells(row: int, col: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Get all 8 adjacent cells (including diagonals) for a given position.

    Parameters
    ----------
    row : int
        Row index (0-based).
    col : int
        Column index (0-based).
    rows : int
        Total number of rows in the grid.
    cols : int
        Total number of columns in the grid.

    Returns
    -------
    List[Tuple[int, int]]
        List of (row, col) tuples for adjacent cells.
    """
    adjacent = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                adjacent.append((new_row, new_col))
    return adjacent


def is_valid_placement(
    grid: List[List[int]],
    row: int,
    col: int,
    regions: List[List[int]],
    rows: int,
    cols: int
) -> bool:
    """
    Check if placing a crown at (row, col) violates any constraints.

    Parameters
    ----------
    grid : List[List[int]]
        Current grid state (0 = empty, 1 = crown).
    row : int
        Row index to check.
    col : int
        Column index to check.
    regions : List[List[int]]
        Region assignments for each cell (same shape as grid).
    rows : int
        Total number of rows.
    cols : int
        Total number of columns.

    Returns
    -------
    bool
        True if placement is valid, False otherwise.
    """
    # Check if cell already has a crown
    if grid[row][col] == 1:
        return False

    # Check row constraint: exactly one crown per row
    row_count = sum(grid[row])
    if row_count >= 1:
        return False

    # Check column constraint: exactly one crown per column
    col_count = sum(grid[r][col] for r in range(rows))
    if col_count >= 1:
        return False

    # Check region constraint: exactly one crown per region
    region_id = regions[row][col]
    region_count = sum(
        1 for r in range(rows) for c in range(cols)
        if regions[r][c] == region_id and grid[r][c] == 1
    )
    if region_count >= 1:
        return False

    # Check adjacency constraint: no adjacent crowns
    adjacent = get_adjacent_cells(row, col, rows, cols)
    for adj_row, adj_col in adjacent:
        if grid[adj_row][adj_col] == 1:
            return False

    return True


def solve_crowns_puzzle(
    rows: int,
    cols: int,
    regions: List[List[int]],
    existing_crowns: Optional[List[Tuple[int, int]]] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Solve a Crowns puzzle using backtracking.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    regions : List[List[int]]
        Region assignments for each cell. regions[r][c] gives the region ID
        for cell at row r, column c.
    existing_crowns : Optional[List[Tuple[int, int]]], optional
        List of (row, col) positions where crowns are already placed.
        Default is None.

    Returns
    -------
    Optional[List[Tuple[int, int]]]
        List of (row, col) positions where crowns should be placed,
        or None if no solution exists.
    """
    # Initialize grid
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Place existing crowns
    if existing_crowns:
        for row, col in existing_crowns:
            if 0 <= row < rows and 0 <= col < cols:
                grid[row][col] = 1

    # Track which rows, columns, and regions have crowns
    rows_with_crowns = set()
    cols_with_crowns = set()
    regions_with_crowns = set()

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                rows_with_crowns.add(row)
                cols_with_crowns.add(col)
                regions_with_crowns.add(regions[row][col])

    # Backtracking solver
    def backtrack(current_row: int) -> bool:
        """
        Recursive backtracking function.

        Parameters
        ----------
        current_row : int
            Current row being processed.

        Returns
        -------
        bool
            True if solution found, False otherwise.
        """
        # Base case: all rows processed
        if current_row >= rows:
            # Verify all constraints are satisfied
            for row in range(rows):
                if sum(grid[row]) != 1:
                    return False
            for col in range(cols):
                if sum(grid[r][col] for r in range(rows)) != 1:
                    return False
            for region_id in set(regions[r][c] for r in range(rows) for c in range(cols)):
                count = sum(
                    1 for r in range(rows) for c in range(cols)
                    if regions[r][c] == region_id and grid[r][c] == 1
                )
                if count != 1:
                    return False
            return True

        # If this row already has a crown, move to next row
        if current_row in rows_with_crowns:
            return backtrack(current_row + 1)

        # Try placing a crown in each column of this row
        for col in range(cols):
            # Skip if column already has a crown
            if col in cols_with_crowns:
                continue

            # Check if placement is valid
            if is_valid_placement(grid, current_row, col, regions, rows, cols):
                # Place crown
                grid[current_row][col] = 1
                rows_with_crowns.add(current_row)
                cols_with_crowns.add(col)
                regions_with_crowns.add(regions[current_row][col])

                # Recursively solve next row
                if backtrack(current_row + 1):
                    return True

                # Backtrack: remove crown
                grid[current_row][col] = 0
                rows_with_crowns.remove(current_row)
                cols_with_crowns.remove(col)
                regions_with_crowns.remove(regions[current_row][col])

        return False

    # Start backtracking from row 0
    if backtrack(0):
        # Extract solution
        solution = []
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    solution.append((row, col))
        return solution

    return None


def validate_solution(
    solution: List[Tuple[int, int]],
    rows: int,
    cols: int,
    regions: List[List[int]]
) -> Tuple[bool, List[str]]:
    """
    Validate that a solution satisfies all constraints.

    Parameters
    ----------
    solution : List[Tuple[int, int]]
        List of (row, col) positions where crowns are placed.
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    regions : List[List[int]]
        Region assignments for each cell.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors) tuple.
    """
    errors = []
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Place crowns
    for row, col in solution:
        if not (0 <= row < rows and 0 <= col < cols):
            errors.append(f"Invalid position: ({row}, {col})")
            return False, errors
        grid[row][col] = 1

    # Check row constraint
    for row in range(rows):
        count = sum(grid[row])
        if count != 1:
            errors.append(f"Row {row} has {count} crowns (expected 1)")

    # Check column constraint
    for col in range(cols):
        count = sum(grid[r][col] for r in range(rows))
        if count != 1:
            errors.append(f"Column {col} has {count} crowns (expected 1)")

    # Check region constraint
    region_counts = {}
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                region_id = regions[row][col]
                region_counts[region_id] = region_counts.get(region_id, 0) + 1

    for region_id, count in region_counts.items():
        if count != 1:
            errors.append(f"Region {region_id} has {count} crowns (expected 1)")

    # Check adjacency constraint
    for row, col in solution:
        adjacent = get_adjacent_cells(row, col, rows, cols)
        for adj_row, adj_col in adjacent:
            if grid[adj_row][adj_col] == 1:
                errors.append(f"Crowns at ({row}, {col}) and ({adj_row}, {adj_col}) are adjacent")

    return len(errors) == 0, errors

