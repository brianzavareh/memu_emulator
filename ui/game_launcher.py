"""
Game Launcher UI - Main application for discovering and running game automations.

This module provides a Tkinter-based GUI that dynamically discovers games
from the game_solutions directory and allows users to run them with a click.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, List
from pathlib import Path

# Ensure project root is in sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.game_discovery import discover_games, load_game_module, get_game_main_function


class GameLauncherUI:
    """
    Main UI application for game launcher.
    
    Provides a graphical interface to discover and run game automation scripts.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the game launcher UI.

        Parameters
        ----------
        root : tk.Tk
            Root Tkinter window
        """
        self.root = root
        self.root.title("Game Automation Launcher")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Track running game thread
        self.running_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Discovered games
        self.games: List[Dict[str, str]] = []
        
        # Setup UI components
        self._setup_ui()
        
        # Initial game discovery
        self.refresh_games()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self):
        """Setup UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header frame with title and refresh button
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(
            header_frame,
            text="Available Games",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        refresh_button = ttk.Button(
            header_frame,
            text="Refresh",
            command=self.refresh_games
        )
        refresh_button.grid(row=0, column=1, padx=(10, 0))
        
        # Games frame with scrollable canvas
        games_container = ttk.Frame(main_frame)
        games_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        games_container.columnconfigure(0, weight=1)
        games_container.rowconfigure(0, weight=1)
        
        # Canvas for scrolling
        canvas = tk.Canvas(games_container, bg="white")
        scrollbar = ttk.Scrollbar(games_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.games_frame = scrollable_frame
        self.canvas = canvas
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # Status text area
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9)
        )
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure status frame row weight
        status_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def _log_status(self, message: str, level: str = "info"):
        """
        Log a message to the status area.

        Parameters
        ----------
        message : str
            Message to log
        level : str, optional
            Log level: "info", "error", "success". Default is "info".
        """
        self.status_text.config(state=tk.NORMAL)
        
        # Color coding
        if level == "error":
            tag = "error"
            self.status_text.tag_config(tag, foreground="red")
        elif level == "success":
            tag = "success"
            self.status_text.tag_config(tag, foreground="green")
        else:
            tag = "info"
        
        # Append message
        self.status_text.insert(tk.END, f"{message}\n", tag)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        # Update UI
        self.root.update_idletasks()
    
    def refresh_games(self):
        """Refresh the list of discovered games."""
        self._log_status("Scanning for games...", "info")
        
        # Discover games
        self.games = discover_games()
        
        # Clear existing buttons
        for widget in self.games_frame.winfo_children():
            widget.destroy()
        
        if not self.games:
            no_games_label = ttk.Label(
                self.games_frame,
                text="No games found. Add games to game_solutions/ directory.",
                foreground="gray"
            )
            no_games_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
            self._log_status("No games found.", "info")
            return
        
        # Create buttons for each game
        for idx, game in enumerate(self.games):
            self._create_game_button(game, idx)
        
        self._log_status(f"Found {len(self.games)} game(s).", "success")
        
        # Update canvas scroll region
        self.canvas.update_idletasks()
    
    def _create_game_button(self, game: Dict[str, str], row: int):
        """
        Create a button for a game.

        Parameters
        ----------
        game : Dict[str, str]
            Game metadata dictionary
        row : int
            Row index for grid layout
        """
        button = ttk.Button(
            self.games_frame,
            text=game['name'],
            command=lambda g=game: self._run_game(g),
            width=30
        )
        button.grid(row=row, column=0, padx=10, pady=5, sticky=(tk.W, tk.E))
        
        # Configure column weight for button expansion
        self.games_frame.columnconfigure(0, weight=1)
    
    def _run_game(self, game: Dict[str, str]):
        """
        Run a game automation script.

        Parameters
        ----------
        game : Dict[str, str]
            Game metadata dictionary
        """
        # Prevent multiple simultaneous executions
        if self.is_running:
            self._log_status(
                f"Another game is already running. Please wait for it to complete.",
                "error"
            )
            return
        
        # Load and validate game module
        self._log_status(f"Loading {game['name']}...", "info")
        module = load_game_module(game['module_path'])
        
        if module is None:
            self._log_status(
                f"Failed to load {game['name']} module.",
                "error"
            )
            return
        
        main_func = get_game_main_function(module)
        if main_func is None:
            self._log_status(
                f"{game['name']} module does not have a main() function.",
                "error"
            )
            return
        
        # Start game in separate thread
        self.is_running = True
        self._log_status(f"Running {game['name']}...", "info")
        
        thread = threading.Thread(
            target=self._execute_game,
            args=(game, main_func),
            daemon=True
        )
        thread.start()
        self.running_thread = thread
    
    def _execute_game(self, game: Dict[str, str], main_func: callable):
        """
        Execute game automation in a separate thread.

        Parameters
        ----------
        game : Dict[str, str]
            Game metadata dictionary
        main_func : callable
            Main function to execute
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute main function
                exit_code = main_func()
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Log output
            if stdout_output:
                self._log_status(f"[{game['name']} Output]\n{stdout_output}", "info")
            if stderr_output:
                self._log_status(f"[{game['name']} Errors]\n{stderr_output}", "error")
            
            # Log completion status
            if exit_code == 0:
                self._log_status(f"{game['name']} completed successfully!", "success")
            else:
                self._log_status(
                    f"{game['name']} completed with exit code {exit_code}.",
                    "error"
                )
        
        except KeyboardInterrupt:
            self._log_status(f"{game['name']} interrupted by user.", "error")
        except Exception as e:
            error_msg = f"Error running {game['name']}: {str(e)}"
            self._log_status(error_msg, "error")
            import traceback
            traceback_str = traceback.format_exc()
            self._log_status(f"Traceback:\n{traceback_str}", "error")
        
        finally:
            self.is_running = False
    
    def _on_closing(self):
        """Handle window closing event."""
        if self.is_running:
            response = messagebox.askyesno(
                "Game Running",
                "A game is currently running. Close anyway?",
                parent=self.root
            )
            if not response:
                return
        
        self.root.destroy()


def main():
    """Main entry point for the game launcher UI."""
    root = tk.Tk()
    app = GameLauncherUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

