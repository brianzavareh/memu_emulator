"""
Binary Sudoku+ board analyzer for extracting grid structure and cell values from screenshots.

This module analyzes screenshots to:
- Detect grid boundaries and cell positions
- Extract cell values (0, 1, or empty) using OCR
- Calculate tap coordinates for each cell
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
import time

# Try to import EasyOCR (best for digit recognition)
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# Try to import pytesseract for OCR fallback
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class BinarySudokuBoardAnalyzer:
    """
    Analyzer for Binary Sudoku+ game boards from screenshots.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the analyzer.

        Parameters
        ----------
        debug : bool, optional
            If True, saves debug images. Default is False.
        """
        self.debug = debug
        # Initialize EasyOCR reader for digit recognition (only digits: 0, 1)
        self.easyocr_reader = None
        if HAS_EASYOCR:
            try:
                # Initialize with English and only digits
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None

    def _detect_grid_lines(
        self,
        grid_region: np.ndarray,
        is_horizontal: bool = True,
        estimated_grid_size: Optional[int] = None
    ) -> List[int]:
        """
        Detect grid lines using OpenCV-recommended morphological operations method.
        
        This method uses MORPH_OPEN to extract horizontal/vertical lines separately,
        which works robustly even with varying line thickness.
        
        Parameters are dynamically adjusted based on grid size.

        Parameters
        ----------
        grid_region : np.ndarray
            Grayscale image of the grid region.
        is_horizontal : bool, optional
            If True, detect horizontal lines (project along rows).
            If False, detect vertical lines (project along columns).
            Default is True.
        estimated_grid_size : Optional[int], optional
            Estimated grid size (e.g., 6, 8, 10). Used to adapt parameters.
            If None, will estimate from grid dimensions.
        
        Returns
        -------
        List[int]
            List of line positions (row indices for horizontal, col indices for vertical).
        """
        # Estimate grid size if not provided
        if estimated_grid_size is None:
            if is_horizontal:
                grid_dimension = grid_region.shape[0]
            else:
                grid_dimension = grid_region.shape[1]
            # Estimate from dimension (assume grid is between 4x4 and 12x12 for Binary Sudoku+)
            # For 6x6 grids, typical cell size is around 30-50 pixels
            typical_cell_size = 30 + (grid_dimension - 120) / 15  # Adjusted for 6x6
            typical_cell_size = max(25, min(50, typical_cell_size))
            estimated_grid_size = int(round(grid_dimension / typical_cell_size))
            estimated_grid_size = max(4, min(12, estimated_grid_size))
        
        # Calculate adaptive parameters based on grid size
        # Binary Sudoku+ has thinner grid lines, so use smaller parameters
        if estimated_grid_size <= 6:
            # 6x6 grid: very thin lines, use minimal block size (3) for maximum sensitivity
            block_size = 3  # Smallest possible block_size to detect very thin lines
            c_value = -3  # Negative C value to be more sensitive to dark lines (thin grid lines)
            kernel_thickness = 1  # Thinner kernel for thin lines
            morph_iterations = 1  # Fewer iterations for thin lines
        elif estimated_grid_size <= 10:
            block_size = max(9, int(grid_region.shape[0] / estimated_grid_size * 0.15))
            if block_size % 2 == 0:
                block_size += 1
            c_value = 2
            kernel_thickness = 1
            morph_iterations = 1
        else:
            block_size = max(7, int(grid_region.shape[0] / estimated_grid_size * 0.1))
            if block_size % 2 == 0:
                block_size += 1
            c_value = 1
            kernel_thickness = 1
            morph_iterations = 1
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "analyzer.py:116",
                    "message": "Adaptive thresholding parameters",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "block_size": block_size,
                        "c_value": c_value,
                        "kernel_thickness": kernel_thickness,
                        "morph_iterations": morph_iterations,
                        "grid_region_shape": list(grid_region.shape)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            grid_region,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value
        )
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "G",
                    "location": "analyzer.py:143",
                    "message": "After adaptive thresholding",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "binary_nonzero": int(np.count_nonzero(binary)),
                        "binary_total": int(binary.size),
                        "binary_max": int(np.max(binary)),
                        "grid_region_mean": float(np.mean(grid_region)),
                        "grid_region_std": float(np.std(grid_region)),
                        "grid_region_min": int(np.min(grid_region)),
                        "grid_region_max": int(np.max(grid_region))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Define structuring elements for morphological operations
        if is_horizontal:
            kernel_length = int(grid_region.shape[1] * 0.6)
            kernel_length = max(25, kernel_length)
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                (kernel_length, kernel_thickness)
            )
            # For 6x6 grids with very thin lines, use MORPH_CLOSE instead of MORPH_OPEN
            # MORPH_CLOSE (dilate then erode) preserves thin lines better than MORPH_OPEN
            if estimated_grid_size <= 6:
                detected_lines = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    horizontal_kernel,
                    iterations=morph_iterations
                )
            else:
                detected_lines = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_OPEN,
                    horizontal_kernel,
                    iterations=morph_iterations
                )
        else:
            kernel_length = int(grid_region.shape[0] * 0.6)
            kernel_length = max(25, kernel_length)
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_thickness, kernel_length)
            )
            # For 6x6 grids with very thin lines, use MORPH_CLOSE instead of MORPH_OPEN
            if estimated_grid_size <= 6:
                detected_lines = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    vertical_kernel,
                    iterations=morph_iterations
                )
            else:
                detected_lines = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_OPEN,
                    vertical_kernel,
                    iterations=morph_iterations
                )
        
        # Project along the appropriate axis
        if is_horizontal:
            projection = np.sum(detected_lines, axis=1)
        else:
            projection = np.sum(detected_lines, axis=0)
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "F",
                    "location": "analyzer.py:177",
                    "message": "After morphological operations",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "detected_lines_nonzero": int(np.count_nonzero(detected_lines)),
                        "detected_lines_total": int(detected_lines.size),
                        "detected_lines_max": int(np.max(detected_lines)),
                        "projection_raw_max": float(np.max(projection)),
                        "projection_raw_mean": float(np.mean(projection)),
                        "projection_raw_sum": float(np.sum(projection))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Normalize projection
        projection = projection.astype(np.float32)
        if projection.max() > 0:
            projection = projection / projection.max()
        
        # Estimate cell size for peak detection
        if is_horizontal:
            grid_dimension = grid_region.shape[0]
        else:
            grid_dimension = grid_region.shape[1]
        
        estimated_cell_size = int(grid_dimension / estimated_grid_size)
        
        # Window size for peak detection (smaller for thin lines)
        if estimated_grid_size <= 6:
            window_size = max(3, estimated_cell_size // 5)  # Smaller window for 6x6
        elif estimated_grid_size <= 10:
            window_size = max(2, estimated_cell_size // 6)
        else:
            window_size = max(2, estimated_cell_size // 7)
        
        # Smooth the projection (use smaller kernel for very thin lines)
        if estimated_grid_size <= 6:
            # For 6x6 grids with very thin lines, use minimal smoothing
            smooth_kernel_size = max(3, min(5, window_size))
        else:
            smooth_kernel_size = max(3, window_size)
        if smooth_kernel_size % 2 == 0:
            smooth_kernel_size += 1
        smooth_kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
        projection_smooth = np.convolve(projection, smooth_kernel, mode='same')
        
        # Calculate threshold (lower threshold for thin lines)
        mean_proj_smooth = np.mean(projection_smooth)
        std_proj_smooth = np.std(projection_smooth)
        max_proj_smooth = np.max(projection_smooth)
        # Use much smaller multiplier for very thin lines (0.1 instead of 0.2 for 6x6)
        if estimated_grid_size <= 6:
            threshold = mean_proj_smooth + 0.1 * std_proj_smooth  # Even smaller for 6x6
        else:
            threshold = mean_proj_smooth + 0.2 * std_proj_smooth
        
        percentile_95 = np.percentile(projection_smooth, 95)
        percentile_90 = np.percentile(projection_smooth, 90)
        percentile_85 = np.percentile(projection_smooth, 85)
        percentile_80 = np.percentile(projection_smooth, 80)
        
        # Use much lower threshold for very thin lines
        # But avoid setting threshold to 0 when percentiles are too low
        if estimated_grid_size <= 6:
            # For 6x6 grids with very thin lines, use even lower thresholds
            # Use percentage of max as fallback if percentiles are too low
            if percentile_80 > 0 and percentile_90 > 0:
                if threshold > percentile_90:
                    threshold = percentile_80 * 0.6  # Much lower for thin lines
                elif threshold > percentile_85:
                    threshold = percentile_80 * 0.7
            else:
                # Fallback: use percentage of max when percentiles are zero or too low
                # For very thin lines, use 1-2% of max to ensure we detect lines
                threshold = max(threshold, max_proj_smooth * 0.01)  # At least 1% of max for 6x6 (very thin lines)
        else:
            # Use lower threshold - try 90th percentile first
            if percentile_90 > 0:
                if threshold > percentile_95:
                    threshold = percentile_90 * 0.8  # Even lower for thin lines
            else:
                # Fallback: use percentage of max when percentiles are zero
                threshold = max(threshold, max_proj_smooth * 0.08)  # At least 8% of max
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:184",
                    "message": "Threshold calculation for grid lines",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "mean_proj": float(mean_proj_smooth),
                        "std_proj": float(std_proj_smooth),
                        "threshold_initial": float(mean_proj_smooth + 0.2 * std_proj_smooth),
                        "threshold_final": float(threshold),
                        "percentile_80": float(percentile_80),
                        "percentile_85": float(percentile_85),
                        "percentile_90": float(percentile_90),
                        "percentile_95": float(percentile_95),
                        "projection_max": float(np.max(projection_smooth)),
                        "projection_min": float(np.min(projection_smooth))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Find local maxima
        line_positions = []
        peaks_found = 0
        peaks_rejected_threshold = 0
        peaks_rejected_maximum = 0
        peaks_rejected_projection = 0
        
        # Much lower threshold check for very thin lines
        # Use projection_smooth for consistency with threshold calculation
        projection_threshold_mult = 0.4 if estimated_grid_size <= 6 else 0.6
        
        # For very thin lines, use a more lenient approach: use percentage of max directly
        threshold_before_1pct = threshold
        threshold_1pct = None
        if estimated_grid_size <= 6:
            # For 6x6 grids with MORPH_CLOSE, we detect more lines but need higher threshold to filter noise
            # Use 5% of max instead of 1% to reduce false positives
            threshold_1pct = max_proj_smooth * 0.05
            # Use the lower threshold, but not too low (5% instead of 1%)
            threshold = min(threshold, threshold_1pct)
        
        # #region agent log
        import json
        import time
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            # Sample some projection values to understand the distribution
            sample_indices = np.linspace(0, len(projection_smooth)-1, min(20, len(projection_smooth)), dtype=int)
            sample_values = [float(projection_smooth[i]) for i in sample_indices]
            
            # Find positions above threshold to understand where peaks might be
            above_threshold_indices = np.where(projection_smooth >= threshold)[0]
            above_threshold_values = [float(projection_smooth[i]) for i in above_threshold_indices[:20]]  # First 20
            
            # For 6x6 grid, check expected line positions (every ~138 pixels for horizontal, ~137 for vertical)
            expected_line_positions = []
            if estimated_grid_size <= 6:
                expected_spacing = grid_dimension / estimated_grid_size
                for i in range(estimated_grid_size + 1):
                    pos = int(i * expected_spacing)
                    if 0 <= pos < len(projection_smooth):
                        expected_line_positions.append({
                            "expected_pos": pos,
                            "actual_value": float(projection_smooth[pos]),
                            "above_threshold": projection_smooth[pos] >= threshold
                        })
            
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H",
                    "location": "analyzer.py:352",
                    "message": "Projection values sample and threshold adjustment",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "threshold_before_1pct": float(threshold_before_1pct),
                        "threshold_1pct": float(threshold_1pct) if threshold_1pct is not None else None,
                        "threshold_final": float(threshold),
                        "projection_max": float(max_proj_smooth),
                        "projection_mean": float(np.mean(projection_smooth)),
                        "projection_std": float(np.std(projection_smooth)),
                        "sample_values": sample_values,
                        "values_above_threshold": int(np.sum(projection_smooth >= threshold)),
                        "above_threshold_indices": [int(i) for i in above_threshold_indices[:20]],
                        "above_threshold_values": above_threshold_values,
                        "expected_line_positions": expected_line_positions
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Check edges first (they might contain peaks)
        edge_peaks_found = 0
        edge_first_val = None
        edge_last_val = None
        edge_first_passed = False
        edge_last_passed = False
        if len(projection_smooth) > 0:
            edge_first_val = float(projection_smooth[0])
            edge_last_val = float(projection_smooth[-1])
            edge_threshold_check = threshold * projection_threshold_mult
            
            # Check first position
            if projection_smooth[0] >= edge_threshold_check:
                edge_first_passed = True
                # Check if it's a local maximum (compare with nearby values)
                is_max = True
                for j in range(1, min(window_size + 1, len(projection_smooth))):
                    if projection_smooth[j] > projection_smooth[0]:
                        is_max = False
                        break
                if is_max:
                    line_positions.append(0)
                    edge_peaks_found += 1
            
            # Check last position
            if len(projection_smooth) > 1 and projection_smooth[-1] >= edge_threshold_check:
                edge_last_passed = True
                # Check if it's a local maximum (compare with nearby values)
                is_max = True
                for j in range(len(projection_smooth) - 2, max(len(projection_smooth) - window_size - 1, -1), -1):
                    if projection_smooth[j] > projection_smooth[-1]:
                        is_max = False
                        break
                if is_max:
                    line_positions.append(len(projection_smooth) - 1)
                    edge_peaks_found += 1
        
        # Check middle positions
        # For very thin lines (6x6), use a more lenient approach: find all positions above threshold
        # and then filter by local maxima with a smaller window
        if estimated_grid_size <= 6:
            # Use smaller window for very thin lines to catch subtle peaks
            peak_window = max(2, window_size // 2)
        else:
            peak_window = window_size
        
        for i in range(peak_window, len(projection_smooth) - peak_window):
            center_val = projection_smooth[i]
            
            if center_val < threshold:
                peaks_rejected_threshold += 1
                continue
            
            is_maximum = True
            for j in range(i - peak_window, i + peak_window + 1):
                if j != i and projection_smooth[j] > center_val:
                    is_maximum = False
                    break
            
            if not is_maximum:
                peaks_rejected_maximum += 1
                continue
            
            # Much lower threshold check for very thin lines
            # Use projection_smooth for consistency with threshold calculation
            if projection_smooth[i] >= threshold * projection_threshold_mult:
                line_positions.append(i)
                peaks_found += 1
            else:
                peaks_rejected_projection += 1
        
        peaks_found += edge_peaks_found
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "analyzer.py:338",
                    "message": "Line detection peak finding results",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "threshold": float(threshold),
                        "projection_threshold_mult": float(projection_threshold_mult),
                        "edge_first_val": edge_first_val,
                        "edge_last_val": edge_last_val,
                        "edge_first_passed": edge_first_passed,
                        "edge_last_passed": edge_last_passed,
                        "edge_peaks_found": edge_peaks_found,
                        "peaks_found": peaks_found,
                        "peaks_rejected_threshold": peaks_rejected_threshold,
                        "peaks_rejected_maximum": peaks_rejected_maximum,
                        "peaks_rejected_projection": peaks_rejected_projection,
                        "line_positions_count": len(line_positions),
                        "line_positions": [int(p) for p in line_positions[:10]]  # First 10 positions
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception as e:
            # Log the exception to understand why logging fails
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B_ERROR",
                        "location": "analyzer.py:338",
                        "message": "Logging error",
                        "data": {"error": str(e)},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except:
                pass
        # #endregion
        
        # Filter out lines that are too close together (smaller spacing for thin lines)
        if estimated_grid_size <= 6:
            min_spacing = max(3, estimated_cell_size // 5)  # Smaller spacing for 6x6 thin lines
        elif estimated_grid_size <= 10:
            min_spacing = max(2, estimated_cell_size // 6)
        else:
            min_spacing = max(2, estimated_cell_size // 7)
        filtered_positions = self._filter_close_values(line_positions, threshold=min_spacing)
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E",
                    "location": "analyzer.py:341",
                    "message": "After filtering close values",
                    "data": {
                        "estimated_grid_size": estimated_grid_size,
                        "is_horizontal": is_horizontal,
                        "line_positions_before_filter": len(line_positions),
                        "line_positions_after_filter": len(filtered_positions),
                        "min_spacing": min_spacing,
                        "estimated_cell_size": estimated_cell_size,
                        "filtered_positions": [int(p) for p in filtered_positions[:15]]  # First 15 positions
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        return filtered_positions

    def _normalize_lines_to_boundaries(
        self, 
        line_positions: List[int], 
        grid_dimension: int, 
        expected_count: int
    ) -> List[int]:
        """
        Normalize detected lines so they are evenly spaced from each other.
        First line is forced to 0, last line is forced to grid_dimension-1.
        Intermediate lines are evenly spaced between them.
        
        Parameters
        ----------
        line_positions : List[int]
            Detected line positions (relative to grid region, 0 to grid_dimension-1)
        grid_dimension : int
            Total dimension of grid (height for horizontal, width for vertical)
        expected_count : int
            Expected number of lines (grid_size + 1)
        
        Returns
        -------
        List[int]
            Normalized line positions evenly spaced from 0 to grid_dimension-1
        """
        if not line_positions or len(line_positions) < 2:
            # If no lines detected, generate evenly spaced lines from boundaries
            return [int(i * (grid_dimension - 1) / (expected_count - 1)) if expected_count > 1 else 0 
                   for i in range(expected_count)]
        
        sorted_lines = sorted(line_positions)
        
        if not sorted_lines or len(sorted_lines) < 2:
            # If no lines detected, generate evenly spaced lines from boundaries
            normalized = [int(i * (grid_dimension - 1) / (expected_count - 1)) if expected_count > 1 else 0 
                         for i in range(expected_count)]
            if normalized:
                normalized[0] = 0
                normalized[-1] = grid_dimension - 1
        else:
            # Use detected first and last lines as boundaries
            first_detected = sorted_lines[0]
            last_detected = sorted_lines[-1]
            
            # Create evenly spaced lines between detected boundaries
            normalized = []
            if expected_count > 1:
                spacing = (last_detected - first_detected) / (expected_count - 1)
                for i in range(expected_count):
                    pos = int(first_detected + i * spacing)
                    normalized.append(pos)
            else:
                normalized.append(first_detected)
            
            # Ensure first line is exactly first_detected and last line is exactly last_detected
            if normalized:
                normalized[0] = first_detected
                normalized[-1] = last_detected
        
        return normalized

    def _filter_close_values(self, values: List[float], threshold: float) -> List[float]:
        """
        Filter out values that are too close together.

        Parameters
        ----------
        values : List[float]
            List of values to filter.
        threshold : float
            Minimum distance between values.

        Returns
        -------
        List[float]
            Filtered list of values.
        """
        if not values:
            return []
        sorted_values = sorted(values)
        filtered = [sorted_values[0]]
        for val in sorted_values[1:]:
            if val - filtered[-1] >= threshold:
                filtered.append(val)
        return filtered

    def detect_grid(
        self,
        screenshot: Image.Image,
        min_grid_size: int = 4,
        max_grid_size: int = 12
    ) -> Optional[Dict[str, Any]]:
        """
        Detect grid boundaries and cell positions from a screenshot.
        
        Uses the approach from the attached file:
        1. Detect grid using Canny edges and contours (find largest square-ish contour)
        2. Fallback to line detection if contour method fails
        3. Detect grid size by counting internal lines
        4. Calculate cell positions based on detected grid size

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot of the game board.
        min_grid_size : int, optional
            Minimum expected grid size (rows/cols). Default is 4.
        max_grid_size : int, optional
            Maximum expected grid size (rows/cols). Default is 12.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - 'rows': Number of rows detected
            - 'cols': Number of columns detected
            - 'cell_width': Width of each cell in pixels
            - 'cell_height': Height of each cell in pixels
            - 'grid_bounds': (x, y, width, height) of grid area
            - 'cell_coordinates': List of (center_x, center_y) for each cell
            Or None if grid detection fails.
        """
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Detect grid using attached file's approach
            grid_img, n, (grid_x, grid_y, grid_width, grid_height) = self._detect_grid_internal(img_cv)
            
            if grid_img is None or n is None:
                print("Error: Could not detect grid in image")
                return None
            
            # Ensure grid size is within expected range
            n = max(min_grid_size, min(max_grid_size, n))
            
            if self.debug:
                print(f"Debug: Detected {n}x{n} grid at ({grid_x}, {grid_y}), size {grid_width}x{grid_height}")
            
            # Calculate cell dimensions
            cell_width = grid_width / n
            cell_height = grid_height / n
            
            # Calculate cell coordinates (centers)
            cell_coordinates = []
            for row in range(n):
                for col in range(n):
                    center_x = int(grid_x + (col + 0.5) * cell_width)
                    center_y = int(grid_y + (row + 0.5) * cell_height)
                    cell_coordinates.append((center_x, center_y))
            
            # Validate cell dimensions
            min_cell_size = 20
            if cell_width < min_cell_size or cell_height < min_cell_size:
                print(f"Warning: Cell size too small ({cell_width}x{cell_height})")
                return None
            
            result = {
                'rows': n,
                'cols': n,
                'cell_width': int(cell_width),
                'cell_height': int(cell_height),
                'grid_bounds': (int(grid_x), int(grid_y), int(grid_width), int(grid_height)),
                'cell_coordinates': cell_coordinates
            }
            
            if self.debug:
                debug_img = img_cv.copy()
                cv2.rectangle(debug_img, (grid_x, grid_y), 
                            (grid_x + grid_width, grid_y + grid_height), (0, 255, 0), 2)
                
                # Draw cell centers
                for center_x, center_y in cell_coordinates:
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                
                cv2.imwrite('debug_binary_sudoku_grid.png', debug_img)
                print(f"Debug: Grid detection complete - {n}x{n}")
            
            return result

        except Exception as e:
            print(f"Error detecting grid: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_grid_internal(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int], Tuple[int, int, int, int]]:
        """
        Detect the sudoku grid and return cropped grid image, size n, and bounding box.
        
        This is the core grid detection logic from the attached file.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format.
        
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[int], Tuple[int, int, int, int]]
            (grid_img, n, (x, y, w, h)) or (None, None, (0, 0, 0, 0)) if detection fails.
        """
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:803",
                    "message": "Entry to _detect_grid_internal",
                    "data": {
                        "image_shape": list(image.shape) if image is not None else None,
                        "image_dtype": str(image.dtype) if image is not None else None,
                        "image_is_none": image is None,
                        "image_size": int(image.size) if image is not None else None
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest square-ish contour (the grid)
        # Filter out small contours - grid should be at least 10% of image area
        min_area_threshold = image.shape[0] * image.shape[1] * 0.1
        max_area = 0
        best_contour = None
        candidate_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Skip very small contours
            if area < min_area_threshold:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # quadrilateral
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:  # roughly square
                    candidate_contours.append((area, contour, (x, y, w, h)))
                    if area > max_area:
                        max_area = area
                        best_contour = contour
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            candidate_info = []
            for area, _, (x, y, w, h) in sorted(candidate_contours, reverse=True)[:5]:  # Top 5
                candidate_info.append({
                    "area": float(area),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "size": f"{int(w)}x{int(h)}"
                })
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:850",
                    "message": "After contour detection",
                    "data": {
                        "num_contours": len(contours),
                        "num_candidates": len(candidate_contours),
                        "min_area_threshold": float(min_area_threshold),
                        "best_contour_found": best_contour is not None,
                        "max_area": float(max_area),
                        "candidates": candidate_info,
                        "edges_nonzero": int(np.count_nonzero(edges)),
                        "edges_total": int(edges.size)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        if best_contour is None:
            # Fallback: find grid by detecting lines
            return self._detect_grid_by_lines(image, gray)
        
        x, y, w, h = cv2.boundingRect(best_contour)
        grid_img = image[y:y+h, x:x+w]
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:870",
                    "message": "After grid_img extraction",
                    "data": {
                        "grid_img_shape": list(grid_img.shape) if grid_img is not None else None,
                        "grid_img_size": int(grid_img.size) if grid_img is not None else None,
                        "grid_img_is_none": grid_img is None,
                        "bounding_box": [int(x), int(y), int(w), int(h)]
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Detect grid size by counting internal lines
        n = self._detect_grid_size(grid_img)
        
        return grid_img, n, (x, y, w, h)
    
    def _detect_grid_by_lines(self, image: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int], Tuple[int, int, int, int]]:
        """
        Fallback method: detect grid by finding horizontal and vertical lines.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format.
        gray : np.ndarray
            Grayscale version of the image.
        
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[int], Tuple[int, int, int, int]]
            (grid_img, n, (x, y, w, h)) or (None, None, (0, 0, 0, 0)) if detection fails.
        """
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:872",
                    "message": "Entry to _detect_grid_by_lines (fallback)",
                    "data": {
                        "image_shape": list(image.shape) if image is not None else None,
                        "gray_shape": list(gray.shape) if gray is not None else None
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine
        grid_lines = cv2.add(horizontal, vertical)
        
        # Find bounding box of the grid
        coords = cv2.findNonZero(grid_lines)
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:905",
                    "message": "After line detection in fallback",
                    "data": {
                        "coords_found": coords is not None,
                        "grid_lines_nonzero": int(np.count_nonzero(grid_lines)),
                        "horizontal_nonzero": int(np.count_nonzero(horizontal)),
                        "vertical_nonzero": int(np.count_nonzero(vertical))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add small padding
            pad = 5
            x, y = max(0, x - pad), max(0, y - pad)
            w, h = min(image.shape[1] - x, w + 2*pad), min(image.shape[0] - y, h + 2*pad)
            grid_img = image[y:y+h, x:x+w]
            n = self._detect_grid_size(grid_img)
            return grid_img, n, (x, y, w, h)
        
        return None, None, (0, 0, 0, 0)
    
    def _detect_grid_size(self, grid_img: np.ndarray) -> int:
        """
        Detect the grid size (n for nxn) by counting grid lines.
        
        Parameters
        ----------
        grid_img : np.ndarray
            Cropped grid image.
        
        Returns
        -------
        int
            Grid size n (for nxn grid).
        
        Raises
        ------
        ValueError
            If grid size cannot be determined.
        """
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "analyzer.py:903",
                    "message": "Entry to _detect_grid_size",
                    "data": {
                        "grid_img_shape": list(grid_img.shape) if grid_img is not None else None,
                        "grid_img_size": int(grid_img.size) if grid_img is not None else None,
                        "grid_img_is_none": grid_img is None
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use Canny edges to detect grid lines
        edges = cv2.Canny(gray, 50, 150)
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "analyzer.py:920",
                    "message": "After Canny edge detection",
                    "data": {
                        "h": int(h),
                        "w": int(w),
                        "edges_nonzero": int(np.count_nonzero(edges)),
                        "edges_total": int(edges.size),
                        "edges_max": int(np.max(edges)),
                        "edges_mean": float(np.mean(edges)),
                        "gray_mean": float(np.mean(gray)),
                        "gray_std": float(np.std(gray))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Detect horizontal lines
        # Use smaller kernel size - w//12 instead of w//4 to avoid over-detection
        # For 823px grid, w//12 ≈ 68px which is more appropriate for grid lines
        kernel_width = max(30, min(w // 12, 100))  # Between 30 and 100 pixels
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Find horizontal line positions
        row_sums = np.sum(horizontal, axis=1)
        threshold = np.max(row_sums) * 0.3
        line_positions = np.where(row_sums > threshold)[0]
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "post-fix",
                    "hypothesisId": "C",
                    "location": "analyzer.py:1077",
                    "message": "After horizontal line detection (fixed kernel)",
                    "data": {
                        "horizontal_kernel_size": [int(kernel_width), 1],
                        "horizontal_nonzero": int(np.count_nonzero(horizontal)),
                        "row_sums_max": float(np.max(row_sums)),
                        "row_sums_mean": float(np.mean(row_sums)),
                        "threshold": float(threshold),
                        "line_positions_count": len(line_positions),
                        "line_positions_sample": [int(p) for p in line_positions[:10]]
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Cluster line positions to count distinct lines
        # Use smaller min_gap initially to avoid merging lines that should be separate
        # For 6x6 grid: cell size ~137px, so min_gap should be < 137px to not merge lines
        min_gap = max(40, h // 20)  # ~41px for 823px, small enough to not merge 137px-spaced lines
        if len(line_positions) > 0:
            horizontal_lines = self._cluster_positions(line_positions, min_gap=min_gap)
            
            # Filter false positives by finding dominant spacing pattern
            if len(horizontal_lines) >= 4:
                # Calculate spacing between consecutive lines
                spacings = [horizontal_lines[i+1] - horizontal_lines[i] 
                        for i in range(len(horizontal_lines) - 1)]
                
                # Find dominant spacing by clustering spacings
                # Group spacings that are within 20% of each other
                spacing_clusters = {}
                for spacing in spacings:
                    # Find existing cluster within 20% tolerance
                    matched = False
                    for cluster_center in spacing_clusters.keys():
                        if abs(spacing - cluster_center) / cluster_center < 0.2:
                            spacing_clusters[cluster_center].append(spacing)
                            matched = True
                            break
                    if not matched:
                        spacing_clusters[spacing] = [spacing]
                
                # Find the largest cluster (dominant spacing)
                if spacing_clusters:
                    dominant_spacing = max(spacing_clusters.keys(), 
                                          key=lambda k: len(spacing_clusters[k]))
                    dominant_spacing_avg = np.mean(spacing_clusters[dominant_spacing])
                    
                    # Filter lines to keep only those that fit the dominant spacing pattern
                    # Keep first line, then only keep lines that are approximately
                    # dominant_spacing apart from the previous kept line
                    filtered_lines = [horizontal_lines[0]]
                    tolerance = dominant_spacing_avg * 0.2  # 20% tolerance
                    
                    for i in range(1, len(horizontal_lines)):
                        gap = horizontal_lines[i] - filtered_lines[-1]
                        # Check if gap is close to dominant spacing or a multiple
                        if abs(gap - dominant_spacing_avg) < tolerance:
                            filtered_lines.append(horizontal_lines[i])
                        elif gap > dominant_spacing_avg * 1.5:
                            # Large gap - might be missing a line, but keep this one
                            # if it's approximately 2x the dominant spacing
                            if abs(gap - 2 * dominant_spacing_avg) < tolerance * 2:
                                filtered_lines.append(horizontal_lines[i])
                    
                    horizontal_lines = filtered_lines
            
            # Infer grid size from spacing if we have enough lines
            if len(horizontal_lines) >= 3:
                # Calculate spacing between consecutive lines
                spacings = [horizontal_lines[i+1] - horizontal_lines[i] 
                           for i in range(len(horizontal_lines) - 1)]
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                
                # If spacing is relatively regular (std < 40% of mean), infer n from spacing
                if std_spacing < 0.4 * avg_spacing and avg_spacing > 0:
                    # Infer grid size: n = h / avg_spacing (rounded)
                    n_from_spacing = int(round(h / avg_spacing))
                    # Validate: we should have approximately n+1 lines
                    expected_lines = n_from_spacing + 1
                    if abs(len(horizontal_lines) - expected_lines) <= 2:
                        # Spacing-based inference is valid
                        n = n_from_spacing
                    else:
                        # Use line count if spacing inference doesn't match
                        n = len(horizontal_lines) - 1
                else:
                    # Spacing is irregular, use line count
                    n = len(horizontal_lines) - 1
            else:
                # Not enough lines, use line count
                n = len(horizontal_lines) - 1 if len(horizontal_lines) > 0 else 0
            
            # #region agent log
            import json
            log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
            try:
                spacing_info = {}
                n_from_spacing_val = None
                dominant_spacing_info = {}
                if len(horizontal_lines) >= 2:
                    spacings = [horizontal_lines[i+1] - horizontal_lines[i] 
                               for i in range(len(horizontal_lines) - 1)]
                    avg_spacing = np.mean(spacings)
                    std_spacing = np.std(spacings)
                    spacing_info = {
                        "avg_spacing": float(avg_spacing),
                        "std_spacing": float(std_spacing),
                        "spacings": [float(s) for s in spacings[:10]]
                    }
                    if len(horizontal_lines) >= 3 and std_spacing < 0.4 * avg_spacing and avg_spacing > 0:
                        n_from_spacing_val = int(round(h / avg_spacing))
                    # Log dominant spacing if we filtered
                    if len(horizontal_lines) >= 4:
                        all_spacings = [horizontal_lines[i+1] - horizontal_lines[i] 
                                       for i in range(len(horizontal_lines) - 1)]
                        spacing_clusters = {}
                        for spacing in all_spacings:
                            matched = False
                            for cluster_center in spacing_clusters.keys():
                                if abs(spacing - cluster_center) / cluster_center < 0.2:
                                    spacing_clusters[cluster_center].append(spacing)
                                    matched = True
                                    break
                            if not matched:
                                spacing_clusters[spacing] = [spacing]
                        if spacing_clusters:
                            dominant_spacing = max(spacing_clusters.keys(), 
                                                  key=lambda k: len(spacing_clusters[k]))
                            dominant_spacing_info = {
                                "dominant_spacing": float(dominant_spacing),
                                "dominant_cluster_size": len(spacing_clusters[dominant_spacing])
                            }
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "post-fix",
                        "hypothesisId": "E",
                        "location": "analyzer.py:1110",
                        "message": "After horizontal clustering (filtered by dominant spacing)",
                        "data": {
                            "horizontal_lines_count": len(horizontal_lines),
                            "n_from_horizontal": int(n),
                            "n_from_spacing": n_from_spacing_val,
                            "min_gap": int(min_gap),
                            "horizontal_lines": [int(l) for l in horizontal_lines[:15]],
                            **spacing_info,
                            **dominant_spacing_info
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if n > 0:
                return n
        
        # Fallback: try vertical lines
        # Use smaller kernel size - h//12 instead of h//4 to avoid over-detection
        kernel_height = max(30, min(h // 12, 100))  # Between 30 and 100 pixels
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        col_sums = np.sum(vertical, axis=0)
        threshold = np.max(col_sums) * 0.3
        line_positions = np.where(col_sums > threshold)[0]
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "post-fix",
                    "hypothesisId": "C",
                    "location": "analyzer.py:1150",
                    "message": "After vertical line detection (fixed kernel)",
                    "data": {
                        "vertical_kernel_size": [1, int(kernel_height)],
                        "vertical_nonzero": int(np.count_nonzero(vertical)),
                        "col_sums_max": float(np.max(col_sums)),
                        "col_sums_mean": float(np.mean(col_sums)),
                        "threshold": float(threshold),
                        "line_positions_count": len(line_positions),
                        "line_positions_sample": [int(p) for p in line_positions[:10]]
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Use smaller min_gap to avoid merging lines
        min_gap = max(40, w // 20)  # ~41px for 823px
        if len(line_positions) > 0:
            vertical_lines = self._cluster_positions(line_positions, min_gap=min_gap)
            
            # Filter false positives by finding dominant spacing pattern
            if len(vertical_lines) >= 4:
                spacings = [vertical_lines[i+1] - vertical_lines[i] 
                           for i in range(len(vertical_lines) - 1)]
                
                # Find dominant spacing by clustering spacings
                spacing_clusters = {}
                for spacing in spacings:
                    matched = False
                    for cluster_center in spacing_clusters.keys():
                        if abs(spacing - cluster_center) / cluster_center < 0.2:
                            spacing_clusters[cluster_center].append(spacing)
                            matched = True
                            break
                    if not matched:
                        spacing_clusters[spacing] = [spacing]
                
                if spacing_clusters:
                    dominant_spacing = max(spacing_clusters.keys(), 
                                          key=lambda k: len(spacing_clusters[k]))
                    dominant_spacing_avg = np.mean(spacing_clusters[dominant_spacing])
                    
                    # Filter lines to keep only those that fit the pattern
                    filtered_lines = [vertical_lines[0]]
                    tolerance = dominant_spacing_avg * 0.2
                    
                    for i in range(1, len(vertical_lines)):
                        gap = vertical_lines[i] - filtered_lines[-1]
                        if abs(gap - dominant_spacing_avg) < tolerance:
                            filtered_lines.append(vertical_lines[i])
                        elif gap > dominant_spacing_avg * 1.5:
                            if abs(gap - 2 * dominant_spacing_avg) < tolerance * 2:
                                filtered_lines.append(vertical_lines[i])
                    
                    vertical_lines = filtered_lines
            
            # Infer grid size from spacing if we have enough lines
            if len(vertical_lines) >= 3:
                spacings = [vertical_lines[i+1] - vertical_lines[i] 
                           for i in range(len(vertical_lines) - 1)]
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                
                # If spacing is relatively regular, infer n from spacing
                if std_spacing < 0.4 * avg_spacing and avg_spacing > 0:
                    n_from_spacing = int(round(w / avg_spacing))
                    expected_lines = n_from_spacing + 1
                    if abs(len(vertical_lines) - expected_lines) <= 2:
                        n = n_from_spacing
                    else:
                        n = len(vertical_lines) - 1
                else:
                    n = len(vertical_lines) - 1
            else:
                n = len(vertical_lines) - 1 if len(vertical_lines) > 0 else 0
            
            # #region agent log
            import json
            log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
            try:
                spacing_info = {}
                if len(vertical_lines) >= 2:
                    spacings = [vertical_lines[i+1] - vertical_lines[i] 
                               for i in range(len(vertical_lines) - 1)]
                    spacing_info = {
                        "avg_spacing": float(np.mean(spacings)),
                        "std_spacing": float(np.std(spacings)),
                        "spacings": [float(s) for s in spacings[:10]]
                    }
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "post-fix",
                        "hypothesisId": "E",
                        "location": "analyzer.py:1180",
                        "message": "After vertical clustering (validated)",
                        "data": {
                            "vertical_lines_count": len(vertical_lines),
                            "n_from_vertical": int(n),
                            "min_gap": int(min_gap),
                            "vertical_lines": [int(l) for l in vertical_lines[:15]],
                            **spacing_info
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if n > 0:
                return n
        
        # #region agent log
        import json
        log_path = r"c:\Users\erfan\Downloads\memu_emulator\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "analyzer.py:1015",
                    "message": "Before raising ValueError - all methods failed",
                    "data": {
                        "h": int(h),
                        "w": int(w),
                        "edges_nonzero": int(np.count_nonzero(edges)),
                        "horizontal_line_positions_count": 0,  # Will be 0 if we reach here
                        "vertical_line_positions_count": len(line_positions) if len(line_positions) > 0 else 0
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        raise ValueError("Could not determine grid size")
    
    def _cluster_positions(self, positions: np.ndarray, min_gap: int) -> List[int]:
        """
        Cluster nearby positions into single line positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of position values.
        min_gap : int
            Minimum gap between clusters.
        
        Returns
        -------
        List[int]
            List of clustered position values.
        """
        if len(positions) == 0:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= min_gap:
                current_cluster.append(pos)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [pos]
        
        clusters.append(int(np.mean(current_cluster)))
        return clusters

    def _read_cell_value(self, cell_img: np.ndarray) -> Optional[int]:
        """
        Read the value (0, 1, or None) from a single cell image.
        
        Uses the approach from the attached file:
        1. Check if cell has significant content (not empty)
        2. Use OCR to read the digit
        3. Fallback to shape analysis if OCR fails
        
        Parameters
        ----------
        cell_img : np.ndarray
            Image of a single cell.
        
        Returns
        -------
        Optional[int]
            Cell value: 0, 1, or None if empty.
        """
        if cell_img.size == 0:
            return None
        
        # Convert to grayscale
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()
        
        # Check if cell has significant content (not empty)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        content_ratio = np.sum(binary > 0) / binary.size
        
        if content_ratio < 0.02:  # Nearly empty cell
            return None
        
        # Use OCR to read the digit
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(cell_img, allowlist='01')
                if results:
                    text = results[0][1].strip()
                    if text == '0':
                        return 0
                    elif text == '1':
                        return 1
            except Exception:
                pass
        
        # Fallback: analyze shape to distinguish 0 from 1
        return self._classify_digit_by_shape(binary)
    
    def _classify_digit_by_shape(self, binary: np.ndarray) -> Optional[int]:
        """
        Classify digit as 0 or 1 based on shape analysis.
        
        Parameters
        ----------
        binary : np.ndarray
            Binary image of the cell.
        
        Returns
        -------
        Optional[int]
            0, 1, or None if classification fails.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        if w < 5 or h < 5:
            return None
        
        aspect_ratio = w / h
        
        # 0 is typically wider/rounder, 1 is tall and thin
        if aspect_ratio < 0.5:
            return 1
        elif aspect_ratio > 0.4:
            # Check for hole in the middle (characteristic of 0)
            roi = binary[y:y+h, x:x+w]
            center_region = roi[h//4:3*h//4, w//4:3*w//4]
            center_fill = np.sum(center_region > 0) / center_region.size if center_region.size > 0 else 1
            
            if center_fill < 0.3:  # Has a hole - likely 0
                return 0
            else:
                return 1
        
        return None

    def extract_cell_values(
        self,
        screenshot: Image.Image,
        grid_info: Dict[str, Any]
    ) -> Optional[List[List[Optional[int]]]]:
        """
        Extract cell values (0, 1, or None for empty) from the screenshot.
        
        Uses the approach from the attached file: get cell boundaries and read each cell.

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot of the game board.
        grid_info : Dict[str, Any]
            Grid information from detect_grid().

        Returns
        -------
        Optional[List[List[Optional[int]]]]
            2D list where grid[row][col] is 0, 1, or None (empty).
            Returns None if extraction fails.
        """
        try:
            rows = grid_info['rows']
            cols = grid_info['cols']
            grid_x, grid_y, grid_width, grid_height = grid_info['grid_bounds']
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            grid_region = img_cv[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
            
            # Get cell boundaries
            boundaries = self._get_cell_boundaries(grid_region, rows)
            
            # Extract values from each cell
            values = []
            for row in range(rows):
                row_values = []
                for col in range(cols):
                    x, y, w, h = boundaries[row][col]
                    # Add padding to avoid grid lines
                    pad_x = int(w * 0.15)
                    pad_y = int(h * 0.15)
                    cell_img = grid_region[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]
                    
                    value = self._read_cell_value(cell_img)
                    row_values.append(value)
                values.append(row_values)
            
            return values

        except Exception as e:
            print(f"Error extracting cell values: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_cell_boundaries(self, grid_img: np.ndarray, n: int) -> List[List[Tuple[int, int, int, int]]]:
        """
        Get the (x, y, w, h) boundaries for each cell in the grid.
        
        Returns a 2D list [row][col] of cell boundaries.
        
        Parameters
        ----------
        grid_img : np.ndarray
            Cropped grid image.
        n : int
            Grid size (nxn).
        
        Returns
        -------
        List[List[Tuple[int, int, int, int]]]
            2D list of (x, y, w, h) boundaries for each cell.
        """
        h, w = grid_img.shape[:2]
        cell_h = h / n
        cell_w = w / n
        
        boundaries = []
        for row in range(n):
            row_boundaries = []
            for col in range(n):
                x = int(col * cell_w)
                y = int(row * cell_h)
                cw = int((col + 1) * cell_w) - x
                ch = int((row + 1) * cell_h) - y
                row_boundaries.append((x, y, cw, ch))
            boundaries.append(row_boundaries)
        
        return boundaries

    def get_cell_coordinates(
        self,
        grid_info: Dict[str, Any]
    ) -> List[List[Tuple[int, int]]]:
        """
        Get center coordinates for each cell in the grid.

        Parameters
        ----------
        grid_info : Dict[str, Any]
            Grid information from detect_grid().

        Returns
        -------
        List[List[Tuple[int, int]]]
            2D list where coordinates[row][col] gives (x, y) center coordinates.
        """
        rows = grid_info['rows']
        cols = grid_info['cols']
        cell_coordinates = grid_info['cell_coordinates']
        grid_x, grid_y, grid_width, grid_height = grid_info['grid_bounds']

        # Organize coordinates into 2D structure
        coords_2d = []
        idx = 0
        for row in range(rows):
            row_coords = []
            for col in range(cols):
                if idx < len(cell_coordinates):
                    row_coords.append(cell_coordinates[idx])
                else:
                    center_x = int(grid_x + (col + 0.5) * (grid_width / cols))
                    center_y = int(grid_y + (row + 0.5) * (grid_height / rows))
                    row_coords.append((center_x, center_y))
                idx += 1
            coords_2d.append(row_coords)

        return coords_2d

    def detect_constraints(
        self,
        screenshot: Image.Image,
        grid_info: Dict[str, Any]
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]:
        """
        Detect constraint markers (= and ≠) between cells.
        
        Uses the approach from the attached file: get cell boundaries and check edge regions.
        
        Parameters
        ----------
        screenshot : Image.Image
            Screenshot of the game board.
        grid_info : Dict[str, Any]
            Grid information from detect_grid().
        
        Returns
        -------
        Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]
            Dictionary mapping cell pairs to constraint type ('=' or '≠').
            Keys are ((row1, col1), (row2, col2)) tuples.
        """
        try:
            constraints = {}
            rows = grid_info['rows']
            cols = grid_info['cols']
            grid_x, grid_y, grid_width, grid_height = grid_info['grid_bounds']
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            grid_region = img_cv[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
            
            # Get cell boundaries
            boundaries = self._get_cell_boundaries(grid_region, rows)
            
            h, w = grid_region.shape[:2]
            cell_h = h / rows
            cell_w = w / cols
            
            # Check horizontal edges (between vertically adjacent cells)
            for row in range(rows - 1):
                for col in range(cols):
                    # Region between cell (row, col) and (row+1, col)
                    x, y, cw, ch = boundaries[row][col]
                    
                    # Edge region at bottom of current cell / top of next cell
                    edge_y = int((row + 1) * cell_h)
                    edge_h = int(cell_h * 0.3)
                    edge_x = x + int(cw * 0.2)
                    edge_w = int(cw * 0.6)
                    
                    y1 = max(0, edge_y - edge_h // 2)
                    y2 = min(h, edge_y + edge_h // 2)
                    
                    if y2 > y1 and edge_w > 0:
                        edge_region = grid_region[y1:y2, edge_x:edge_x+edge_w]
                        constraint = self._detect_constraint_in_region(edge_region)
                        if constraint:
                            constraints[((row, col), (row + 1, col))] = constraint
            
            # Check vertical edges (between horizontally adjacent cells)
            for row in range(rows):
                for col in range(cols - 1):
                    x, y, cw, ch = boundaries[row][col]
                    
                    # Edge region at right of current cell / left of next cell
                    edge_x = int((col + 1) * cell_w)
                    edge_w = int(cell_w * 0.3)
                    edge_y = y + int(ch * 0.2)
                    edge_h = int(ch * 0.6)
                    
                    x1 = max(0, edge_x - edge_w // 2)
                    x2 = min(w, edge_x + edge_w // 2)
                    
                    if x2 > x1 and edge_h > 0:
                        edge_region = grid_region[edge_y:edge_y+edge_h, x1:x2]
                        constraint = self._detect_constraint_in_region(edge_region)
                        if constraint:
                            constraints[((row, col), (row, col + 1))] = constraint
            
            return constraints
            
        except Exception as e:
            if self.debug:
                print(f"Error detecting constraints: {e}")
                import traceback
                traceback.print_exc()
            return {}
    
    def _detect_constraint_in_region(self, region: np.ndarray) -> Optional[str]:
        """
        Detect if a region contains = or ≠ symbol.
        
        Parameters
        ----------
        region : np.ndarray
            Image region to analyze.
        
        Returns
        -------
        Optional[str]
            '=' if equals sign detected, '≠' if not-equals detected, None otherwise.
        """
        if region.size == 0:
            return None
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Check for content
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        content_ratio = np.sum(binary > 0) / binary.size
        
        if content_ratio < 0.01:
            return None
        
        # Use OCR to detect = or ≠
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(region, allowlist='=≠+-x/!')
                if results:
                    text = results[0][1].strip().lower()
                    if '=' in text and ('≠' in text or '!' in text or 'x' in text or '/' in text):
                        return '≠'
                    elif '=' in text:
                        return '='
            except Exception:
                pass
        
        # Fallback: analyze horizontal lines
        return self._classify_constraint_by_lines(binary)
    
    def _classify_constraint_by_lines(self, binary: np.ndarray) -> Optional[str]:
        """
        Classify constraint symbol by analyzing horizontal line patterns.
        
        Parameters
        ----------
        binary : np.ndarray
            Binary image of the constraint region.
        
        Returns
        -------
        Optional[str]
            '=' if equals sign detected, '≠' if not-equals detected, None otherwise.
        """
        h, w = binary.shape
        if h < 3 or w < 3:
            return None
        
        # Look for horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal line segments
        row_sums = np.sum(horizontal, axis=1)
        threshold = np.max(row_sums) * 0.3 if np.max(row_sums) > 0 else 0
        
        line_rows = np.where(row_sums > threshold)[0]
        
        if len(line_rows) == 0:
            return None
        
        # Cluster the line rows
        if len(line_rows) > 0:
            clusters = []
            current_cluster = [line_rows[0]]
            
            for pos in line_rows[1:]:
                if pos - current_cluster[-1] <= h // 6:
                    current_cluster.append(pos)
                else:
                    clusters.append(int(np.mean(current_cluster)))
                    current_cluster = [pos]
            
            clusters.append(int(np.mean(current_cluster)))
            lines = clusters
        else:
            lines = []
        
        if len(lines) >= 2:
            # Check for diagonal strike-through (≠)
            # Look for diagonal content between the two lines
            if len(lines) == 2:
                mid_region = binary[lines[0]:lines[1], :]
                # Check for diagonal pattern
                diag_content = np.sum(mid_region) / mid_region.size if mid_region.size > 0 else 0
                if diag_content > 0.05:
                    return '≠'
            return '='
        
        return None

