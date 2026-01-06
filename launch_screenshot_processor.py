"""
Launcher script for the Interactive Screenshot Processor UI.

Run this script to start the graphical user interface for processing
BlueStacks VM screenshots with OpenCV filtering and OCR operations.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.screenshot_processor import main

if __name__ == "__main__":
    main()

