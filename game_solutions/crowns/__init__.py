"""
Crowns game automation package.

This package provides tools for automatically solving the Crowns puzzle game:
- solver: Constraint satisfaction algorithm to solve puzzles
- analyzer: Screenshot analysis to extract grid and regions
- automation: Main automation script to play the game
"""

from .solver import solve_crowns_puzzle
from .analyzer import CrownsBoardAnalyzer

__all__ = ['solve_crowns_puzzle', 'CrownsBoardAnalyzer']

