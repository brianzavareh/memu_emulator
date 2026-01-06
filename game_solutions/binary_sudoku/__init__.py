"""
Binary Sudoku+ game automation package.

This package provides tools for automatically solving the Binary Sudoku+ puzzle game:
- solver: Constraint satisfaction algorithm to solve puzzles
- analyzer: Screenshot analysis to extract grid and cell values
- automation: Main automation script to play the game
"""

from .solver import solve_binary_sudoku_puzzle
from .analyzer import BinarySudokuBoardAnalyzer

__all__ = ['solve_binary_sudoku_puzzle', 'BinarySudokuBoardAnalyzer']

