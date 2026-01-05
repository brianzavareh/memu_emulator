"""
Launcher script for the Game Automation Launcher UI.

Run this script to start the graphical user interface for discovering
and running game automation scripts.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.game_launcher import main

if __name__ == "__main__":
    main()

