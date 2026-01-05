"""
Game discovery module for dynamically finding and loading game automation modules.

This module scans the game_solutions directory for game subdirectories and
dynamically imports their automation modules.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def format_game_name(directory_name: str) -> str:
    """
    Format directory name into a display-friendly game name.

    Parameters
    ----------
    directory_name : str
        Raw directory name (e.g., "crowns", "my_game")

    Returns
    -------
    str
        Formatted game name (e.g., "Crowns", "My Game")
    """
    # Replace underscores and hyphens with spaces
    name = directory_name.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in name.split())


def discover_games(game_solutions_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Discover all games in the game_solutions directory.

    Scans the game_solutions directory for subdirectories containing
    automation.py files and returns metadata about each discovered game.

    Parameters
    ----------
    game_solutions_dir : Optional[Path], optional
        Path to game_solutions directory. If None, uses project root.
        Default is None.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing game metadata:
        - name: Display name of the game
        - directory: Directory name
        - module_path: Full module path (e.g., "game_solutions.crowns.automation")
        - file_path: Full path to automation.py file
    """
    if game_solutions_dir is None:
        # Get project root (assuming this file is in ui/ directory)
        project_root = Path(__file__).parent.parent
        game_solutions_dir = project_root / "game_solutions"
        
        # Ensure project root is in sys.path for imports
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
    else:
        game_solutions_dir = Path(game_solutions_dir)

    if not game_solutions_dir.exists():
        return []

    games = []
    
    # Scan all subdirectories in game_solutions
    for item in game_solutions_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Skip __pycache__ and other hidden/system directories
        if item.name.startswith('_') or item.name.startswith('.'):
            continue
        
        # Check if automation.py exists
        automation_file = item / "automation.py"
        if not automation_file.exists():
            continue
        
        # Format game name
        display_name = format_game_name(item.name)
        
        # Build module path
        module_path = f"game_solutions.{item.name}.automation"
        
        games.append({
            'name': display_name,
            'directory': item.name,
            'module_path': module_path,
            'file_path': str(automation_file)
        })
    
    # Sort by name for consistent display
    games.sort(key=lambda x: x['name'])
    
    return games


def load_game_module(module_path: str) -> Optional[object]:
    """
    Dynamically load a game automation module.

    Parameters
    ----------
    module_path : str
        Full module path (e.g., "game_solutions.crowns.automation")

    Returns
    -------
    Optional[object]
        Loaded module object, or None if loading failed.
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        return module
    except ImportError as e:
        print(f"Error importing module {module_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading module {module_path}: {e}")
        return None


def get_game_main_function(module: object) -> Optional[callable]:
    """
    Get the main function from a game automation module.

    Parameters
    ----------
    module : object
        Loaded module object

    Returns
    -------
    Optional[callable]
        Main function if found, None otherwise.
    """
    if not hasattr(module, 'main'):
        return None
    
    main_func = getattr(module, 'main')
    if not callable(main_func):
        return None
    
    return main_func

