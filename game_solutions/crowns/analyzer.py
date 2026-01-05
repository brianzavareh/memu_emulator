"""
Crowns board analyzer for extracting grid structure and regions from screenshots.

This module analyzes screenshots to:
- Detect grid boundaries and cell positions
- Identify regions by background color
- Calculate tap coordinates for each cell
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict


class CrownsBoardAnalyzer:
    """
    Analyzer for Crowns game boards from screenshots.
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

    def _detect_grid_lines(
        self,
        grid_region: np.ndarray,
        is_horizontal: bool = True
    ) -> List[int]:
        """
        Detect grid lines using OpenCV-recommended morphological operations method.
        
        This method uses MORPH_OPEN to extract horizontal/vertical lines separately,
        which works robustly even with varying line thickness (thick region borders
        and thin internal grid lines).
        
        Based on OpenCV tutorial: Morphological Line Detection
        https://docs.opencv.org/master/dd/dd7/tutorial_morph_lines_detection.html
        
        Parameters
        ----------
        grid_region : np.ndarray
            Grayscale image of the grid region.
        is_horizontal : bool, optional
            If True, detect horizontal lines (project along rows).
            If False, detect vertical lines (project along columns).
            Default is True.
        
        Returns
        -------
        List[int]
            List of line positions (row indices for horizontal, col indices for vertical).
        """
        # Step 1: Apply adaptive thresholding (handles varying lighting)
        # This is better than Otsu for grids with varying backgrounds
        binary = cv2.adaptiveThreshold(
            grid_region,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so dark lines become white
            15,  # Block size - should be odd, adjust based on cell size
            2    # C constant subtracted from mean
        )
        
        # Step 2: Define structuring elements for morphological operations
        # The key is to use LONG kernels in the direction we want to extract
        # For horizontal lines: kernel is wide (many pixels) but only 1 pixel tall
        # For vertical lines: kernel is tall (many pixels) but only 1 pixel wide
        # This extracts lines regardless of their thickness!
        
        # Calculate kernel length based on grid size
        # Should be long enough to span multiple cells but not too long
        if is_horizontal:
            # Horizontal kernel: wide enough to span most of the grid width
            kernel_length = int(grid_region.shape[1] * 0.6)  # 60% of width
            kernel_length = max(25, kernel_length)  # At least 25 pixels
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                (kernel_length, 1)  # Wide but only 1 pixel tall
            )
            
            # Use MORPH_OPEN to extract horizontal lines
            # MORPH_OPEN = erosion followed by dilation
            # This removes small objects and isolates horizontal lines
            detected_lines = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                horizontal_kernel,
                iterations=2  # Apply multiple times for better extraction
            )
        else:
            # Vertical kernel: tall enough to span most of the grid height
            kernel_length = int(grid_region.shape[0] * 0.6)  # 60% of height
            kernel_length = max(25, kernel_length)  # At least 25 pixels
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, kernel_length)  # Tall but only 1 pixel wide
            )
            
            # Use MORPH_OPEN to extract vertical lines
            detected_lines = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                vertical_kernel,
                iterations=2  # Apply multiple times for better extraction
            )
        
        # Step 3: Project along the appropriate axis to find line positions
        # Sum along columns for horizontal lines, along rows for vertical lines
        if is_horizontal:
            projection = np.sum(detected_lines, axis=1)
        else:
            projection = np.sum(detected_lines, axis=0)
        
        # Step 4: Normalize projection to 0-1 range
        projection = projection.astype(np.float32)
        if projection.max() > 0:
            projection = projection / projection.max()
        
        # Step 5: Find peaks in the projection
        # Lines will show up as peaks in the projection
        # Note: We'll calculate threshold AFTER smoothing, as smoothing affects peak values
        
        
        # Estimate expected cell size for peak detection window
        # Try to detect grid size first by looking at spacing
        if is_horizontal:
            grid_dimension = grid_region.shape[0]
        else:
            grid_dimension = grid_region.shape[1]
        
        # Estimate cell size (assume grid is between 3x3 and 15x15)
        # Use average of possible cell sizes
        possible_sizes = list(range(3, 16))
        estimated_cell_size = int(np.mean([grid_dimension / size for size in possible_sizes]))
        
        # Window size for peak detection should be smaller than cell size
        window_size = max(3, estimated_cell_size // 4)
        
        # Smooth the projection to reduce noise
        smooth_kernel_size = max(3, window_size)
        if smooth_kernel_size % 2 == 0:
            smooth_kernel_size += 1  # Make it odd
        smooth_kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
        projection_smooth = np.convolve(projection, smooth_kernel, mode='same')
        
        # Calculate threshold based on SMOOTHED projection statistics
        # This is critical because smoothing reduces peak values
        mean_proj_smooth = np.mean(projection_smooth)
        std_proj_smooth = np.std(projection_smooth)
        
        # Use a lower multiplier for smoothed projection (smoothing reduces variance)
        # Also ensure threshold is not too high - use 1.0 * std instead of 1.5
        threshold = mean_proj_smooth + 1.0 * std_proj_smooth
        
        # Fallback: if threshold is still too high, use percentile-based threshold
        # Ensure at least some peaks can be detected
        percentile_95 = np.percentile(projection_smooth, 95)
        if threshold > percentile_95:
            threshold = percentile_95 * 0.9  # Use 90% of 95th percentile
        
        # Find local maxima (peaks) that exceed threshold
        line_positions = []
        
        for i in range(window_size, len(projection_smooth) - window_size):
            center_val = projection_smooth[i]
            
            # Must exceed threshold
            if center_val < threshold:
                continue
            
            # Check if it's a local maximum
            is_maximum = True
            for j in range(i - window_size, i + window_size + 1):
                if j != i and projection_smooth[j] > center_val:
                    is_maximum = False
                    break
            
            if is_maximum:
                # Verify with original (unsmoothed) projection
                if projection[i] >= threshold * 0.8:  # Allow some tolerance
                    line_positions.append(i)
        
        # Step 6: Filter out lines that are too close together
        # Minimum spacing should be at least 1/4 of estimated cell size
        min_spacing = max(3, estimated_cell_size // 4)
        filtered_positions = self._filter_close_values(line_positions, threshold=min_spacing)
        
        return filtered_positions
    
    def _detect_grid_lines_hough(
        self,
        grid_region: np.ndarray,
        is_horizontal: bool = True
    ) -> List[int]:
        """
        Fallback method: Detect lines using Hough Line Transform.
        
        Parameters
        ----------
        grid_region : np.ndarray
            Grayscale image of the grid region.
        is_horizontal : bool, optional
            If True, detect horizontal lines.
            If False, detect vertical lines.
            Default is True.
        
        Returns
        -------
        List[int]
            List of line positions.
        """
        # Preprocess
        blurred = cv2.GaussianBlur(grid_region, (5, 5), 0)
        
        # Edge detection
        median_val = np.median(blurred)
        lower_threshold = int(max(0, 0.7 * median_val))
        upper_threshold = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        
        # Morphological operations
        if is_horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_region.shape[1] // 2, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grid_region.shape[0] // 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Hough Line Transform
        if is_horizontal:
            min_line_length = int(grid_region.shape[1] * 0.7)
        else:
            min_line_length = int(grid_region.shape[0] * 0.7)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=20
        )
        
        if lines is None or len(lines) == 0:
            return []
        
        # Extract positions
        line_positions = []
        angle_tolerance = 5 * np.pi / 180
        target_angle = 0.0 if is_horizontal else np.pi / 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 == x1:
                angle = np.pi / 2
            else:
                angle = np.arctan2(y2 - y1, x2 - x1)
            
            if angle < 0:
                angle += np.pi
            
            angle_diff = abs(angle - target_angle)
            if angle_diff > np.pi / 2:
                angle_diff = np.pi - angle_diff
            
            if angle_diff <= angle_tolerance:
                if is_horizontal:
                    pos = int((y1 + y2) / 2)
                else:
                    pos = int((x1 + x2) / 2)
                line_positions.append(pos)
        
        # Cluster nearby lines
        if not line_positions:
            return []
        
        line_positions = sorted(line_positions)
        clustered = []
        current_cluster = [line_positions[0]]
        
        for pos in line_positions[1:]:
            if pos - current_cluster[-1] <= 5:
                current_cluster.append(pos)
            else:
                clustered.append(int(np.mean(current_cluster)))
                current_cluster = [pos]
        
        if current_cluster:
            clustered.append(int(np.mean(current_cluster)))
        
        min_spacing = max(5, (grid_region.shape[0] if is_horizontal else grid_region.shape[1]) // 20)
        return self._filter_close_values(clustered, threshold=min_spacing)

    def detect_grid(
        self,
        screenshot: Image.Image,
        min_grid_size: int = 3,
        max_grid_size: int = 15
    ) -> Optional[Dict[str, Any]]:
        """
        Detect grid boundaries and cell positions from a screenshot.
        
        Approach:
        1. Detect the grid as a large square/rectangle (fixed bounds)
        2. Extract the grid region and detect dark gray vertical/horizontal lines
        3. Count lines to determine actual grid size
        4. Calculate cell positions based on detected grid size

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot of the game board.
        min_grid_size : int, optional
            Minimum expected grid size (rows/cols). Default is 3.
        max_grid_size : int, optional
            Maximum expected grid size (rows/cols). Default is 15.

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
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Step 1: Use exact grid coordinates from user feedback
            # Grid is at (16, 166) to (884, 1035) - a square grid
            img_height, img_width = img_cv.shape[:2]
            
            # Use the exact coordinates provided by user
            grid_x = 16
            grid_y = 166
            grid_width = 868  # 884 - 16
            grid_height = 869  # 1035 - 166
            
            # Make it perfectly square (use the smaller dimension for consistency)
            if grid_width != grid_height:
                # Use the average or smaller dimension
                grid_size = min(grid_width, grid_height)
                # Center the square
                grid_x = 16 + (grid_width - grid_size) // 2
                grid_y = 166 + (grid_height - grid_size) // 2
                grid_width = grid_size
                grid_height = grid_size
            
            # Step 2: Extract grid region and detect dark gray lines
            # Ensure coordinates are within image bounds
            grid_x = max(0, int(grid_x))
            grid_y = max(0, int(grid_y))
            grid_width = min(int(grid_width), img_width - grid_x)
            grid_height = min(int(grid_height), img_height - grid_y)
            
            grid_region = gray[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
            
            if grid_region.size == 0:
                print("Error: Grid region is empty")
                return None
            
            # Detect horizontal lines (rows)
            horizontal_lines = self._detect_grid_lines(grid_region, is_horizontal=True)
            # Detect vertical lines (columns)
            vertical_lines = self._detect_grid_lines(grid_region, is_horizontal=False)
            
            if self.debug:
                print(f"Debug: Initially detected {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
                print(f"Debug: Horizontal lines: {horizontal_lines[:10] if len(horizontal_lines) > 10 else horizontal_lines}")
                print(f"Debug: Vertical lines: {vertical_lines[:10] if len(vertical_lines) > 10 else vertical_lines}")
            
            # Grid size = number of lines - 1 (lines separate cells)
            # For a grid with N cells, there are N+1 lines (including borders)
            # So if we detect L lines, we have L-1 cells
            # However, we might not detect border lines, so we need to be careful
            
            # Filter lines to get the main grid pattern
            # Use clustering to find the dominant spacing pattern
            if len(horizontal_lines) >= 2:
                horizontal_lines = self._cluster_and_filter_lines(
                    horizontal_lines, min_grid_size, max_grid_size
                )
            if len(vertical_lines) >= 2:
                vertical_lines = self._cluster_and_filter_lines(
                    vertical_lines, min_grid_size, max_grid_size
                )
            
            if self.debug:
                print(f"Debug: After filtering: {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
            
            # Check if we have enough lines to detect grid
            if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
                print(f"Error: Not enough lines detected. Horizontal: {len(horizontal_lines)}, Vertical: {len(vertical_lines)}")
                print("Cannot determine grid size without sufficient line detection.")
                return None
            
            # Sort lines to ensure proper order
            horizontal_lines_sorted = sorted(horizontal_lines)
            vertical_lines_sorted = sorted(vertical_lines)
            
            # Calculate spacing between consecutive lines to determine if borders are missing
            if len(horizontal_lines_sorted) > 1:
                horizontal_spacings = [
                    horizontal_lines_sorted[i+1] - horizontal_lines_sorted[i]
                    for i in range(len(horizontal_lines_sorted) - 1)
                ]
                avg_h_spacing = np.mean(horizontal_spacings)
                min_h_spacing = np.min(horizontal_spacings)
                max_h_spacing = np.max(horizontal_spacings)
            else:
                avg_h_spacing = grid_height / 10  # Rough estimate
                min_h_spacing = avg_h_spacing
                max_h_spacing = avg_h_spacing
            
            if len(vertical_lines_sorted) > 1:
                vertical_spacings = [
                    vertical_lines_sorted[i+1] - vertical_lines_sorted[i]
                    for i in range(len(vertical_lines_sorted) - 1)
                ]
                avg_v_spacing = np.mean(vertical_spacings)
                min_v_spacing = np.min(vertical_spacings)
                max_v_spacing = np.max(vertical_spacings)
            else:
                avg_v_spacing = grid_width / 10  # Rough estimate
                min_v_spacing = avg_v_spacing
                max_v_spacing = avg_v_spacing
            
            # Check if we're missing border lines at the edges
            # If first line is far from 0 (more than half average spacing), add border line
            # If last line is far from grid_size-1 (more than half average spacing), add border line
            if len(horizontal_lines_sorted) > 0:
                if horizontal_lines_sorted[0] > avg_h_spacing * 0.5:
                    # Missing border line at top - add it at position 0
                    horizontal_lines_sorted.insert(0, 0)
                if horizontal_lines_sorted[-1] < grid_height - avg_h_spacing * 0.5:
                    # Missing border line at bottom - add it at grid_height - 1
                    horizontal_lines_sorted.append(grid_height - 1)
            
            if len(vertical_lines_sorted) > 0:
                if vertical_lines_sorted[0] > avg_v_spacing * 0.5:
                    # Missing border line at left - add it at position 0
                    vertical_lines_sorted.insert(0, 0)
                if vertical_lines_sorted[-1] < grid_width - avg_v_spacing * 0.5:
                    # Missing border line at right - add it at grid_width - 1
                    vertical_lines_sorted.append(grid_width - 1)
            
            if self.debug:
                print(f"Debug: Horizontal spacing - avg: {avg_h_spacing:.1f}, min: {min_h_spacing:.1f}, max: {max_h_spacing:.1f}")
                print(f"Debug: Vertical spacing - avg: {avg_v_spacing:.1f}, min: {min_v_spacing:.1f}, max: {max_v_spacing:.1f}")
            
            # Estimate grid size from line count
            # If spacing is relatively uniform, we likely detected all lines (L lines = L-1 cells)
            # If spacing varies significantly, we might have missed some lines
            spacing_variance_h = np.std(horizontal_spacings) / avg_h_spacing if len(horizontal_spacings) > 0 else 1.0
            spacing_variance_v = np.std(vertical_spacings) / avg_v_spacing if len(vertical_spacings) > 0 else 1.0
            
            # Grid is always n×n square, so calculate size once and use for both dimensions
            # If variance is low (< 0.3), spacing is uniform, so L lines = L-1 cells
            # If variance is high, we might have missed lines - try L lines = L cells
            
            # Calculate estimates for both dimensions
            if spacing_variance_h < 0.3:
                estimated_rows_from_h = len(horizontal_lines_sorted) - 1
            else:
                # Spacing varies - might have missed some lines
                # Estimate from grid height and average spacing
                estimated_rows_from_h = int(round(grid_height / avg_h_spacing))
            
            if spacing_variance_v < 0.3:
                estimated_cols_from_v = len(vertical_lines_sorted) - 1
            else:
                # Spacing varies - might have missed lines
                # Estimate from grid width and average spacing
                estimated_cols_from_v = int(round(grid_width / avg_v_spacing))
            
            # Since grid is always square, use the average or the more reliable estimate
            # Prefer the one with lower variance (more reliable)
            if spacing_variance_h <= spacing_variance_v:
                # Horizontal detection is more reliable
                estimated_size = estimated_rows_from_h
            else:
                # Vertical detection is more reliable
                estimated_size = estimated_cols_from_v
            
            # Ensure both dimensions are the same (square grid)
            estimated_rows = estimated_size
            estimated_cols = estimated_size
            
            if self.debug:
                print(f"Debug: Estimated grid size (square): {estimated_rows}x{estimated_cols}")
                print(f"Debug: Spacing variance - H: {spacing_variance_h:.3f}, V: {spacing_variance_v:.3f}")
                print(f"Debug: Raw estimates - rows: {estimated_rows_from_h}, cols: {estimated_cols_from_v}")
            
            # Validate grid size
            if estimated_size < min_grid_size or estimated_size > max_grid_size:
                print(f"Error: Detected {estimated_size}x{estimated_size} grid, which is outside valid range [{min_grid_size}, {max_grid_size}]")
                return None
            
            # Use detected lines to calculate cell positions directly
            # This is more accurate than assuming uniform spacing
            # We need to ensure we have enough lines for the detected grid size
            if len(horizontal_lines_sorted) >= estimated_rows + 1 and len(vertical_lines_sorted) >= estimated_cols + 1:
                # We have enough lines - use them directly
                # Calculate average cell dimensions from detected lines first
                if estimated_rows > 0 and len(horizontal_lines_sorted) >= 2:
                    total_h_spacing = horizontal_lines_sorted[-1] - horizontal_lines_sorted[0]
                    cell_height = total_h_spacing / estimated_rows
                else:
                    cell_height = grid_height / estimated_rows
                
                if estimated_cols > 0 and len(vertical_lines_sorted) >= 2:
                    total_v_spacing = vertical_lines_sorted[-1] - vertical_lines_sorted[0]
                    cell_width = total_v_spacing / estimated_cols
                else:
                    cell_width = grid_width / estimated_cols
                
                # Now calculate cell positions using detected lines
                cell_coordinates = []
                for row_idx in range(estimated_rows):
                    if row_idx + 1 < len(horizontal_lines_sorted):
                        row_top = horizontal_lines_sorted[row_idx]
                        row_bottom = horizontal_lines_sorted[row_idx + 1]
                        row_center_y = grid_y + (row_top + row_bottom) / 2
                    else:
                        # Fallback: use uniform spacing for last row
                        row_center_y = grid_y + (row_idx + 0.5) * cell_height
                    
                    for col_idx in range(estimated_cols):
                        if col_idx + 1 < len(vertical_lines_sorted):
                            col_left = vertical_lines_sorted[col_idx]
                            col_right = vertical_lines_sorted[col_idx + 1]
                            col_center_x = grid_x + (col_left + col_right) / 2
                        else:
                            # Fallback: use uniform spacing for last col
                            col_center_x = grid_x + (col_idx + 0.5) * cell_width
                        
                        cell_coordinates.append((int(col_center_x), int(row_center_y)))
            else:
                # Not enough lines detected - use uniform spacing as fallback
                cell_width = grid_width / estimated_cols
                cell_height = grid_height / estimated_rows
                
                # Calculate cell center coordinates using uniform spacing
                cell_coordinates = []
                for row in range(estimated_rows):
                    for col in range(estimated_cols):
                        center_x = grid_x + (col + 0.5) * cell_width
                        center_y = grid_y + (row + 0.5) * cell_height
                        cell_coordinates.append((int(center_x), int(center_y)))
            
            # Validate cell dimensions
            min_cell_size = 20
            if cell_width < min_cell_size or cell_height < min_cell_size:
                print(f"Warning: Cell size too small ({cell_width}x{cell_height})")
                return None
            
            result = {
                'rows': estimated_rows,
                'cols': estimated_cols,
                'cell_width': int(cell_width),
                'cell_height': int(cell_height),
                'grid_bounds': (int(grid_x), int(grid_y), int(grid_width), int(grid_height)),
                'cell_coordinates': cell_coordinates
            }
            
            if self.debug:
                # Draw detected grid for debugging
                debug_img = img_cv.copy()
                cv2.rectangle(debug_img, (grid_x, grid_y), 
                            (grid_x + grid_width, grid_y + grid_height), (0, 255, 0), 2)
                
                # Draw detected lines
                for line_y in horizontal_lines:
                    cv2.line(debug_img, (grid_x, grid_y + line_y), 
                           (grid_x + grid_width, grid_y + line_y), (255, 0, 0), 1)
                for line_x in vertical_lines:
                    cv2.line(debug_img, (grid_x + line_x, grid_y), 
                           (grid_x + line_x, grid_y + grid_height), (255, 0, 0), 1)
                
                # Draw cell centers
                for center_x, center_y in cell_coordinates:
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                
                cv2.imwrite('debug_grid.png', debug_img)
                print(f"Debug: Detected {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
                print(f"Debug: Grid size = {estimated_rows}x{estimated_cols}")
            
            return result

        except Exception as e:
            print(f"Error detecting grid: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_cell_data(
        self,
        screenshot: Image.Image,
        grid_info: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cell data: coordinates, (row, col), and median color for each cell.
        
        This returns a simple list where each element contains:
        - 'row': row index (0-based)
        - 'col': column index (0-based)
        - 'center_x': x coordinate of cell center
        - 'center_y': y coordinate of cell center
        - 'color': median RGB color of the cell center

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot of the game board.
        grid_info : Dict[str, Any]
            Grid information from detect_grid().

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            List of cell data dictionaries, or None if detection fails.
        """
        try:
            rows = grid_info['rows']
            cols = grid_info['cols']
            cell_coordinates = grid_info['cell_coordinates']
            cell_width = grid_info['cell_width']
            cell_height = grid_info['cell_height']
            grid_x, grid_y, grid_width, grid_height = grid_info['grid_bounds']

            # Sample the exact center color of each cell (optimized)
            img_array = np.array(screenshot)
            num_cells = len(cell_coordinates)
            cell_colors = [None] * num_cells  # Pre-allocate to maintain order
            sample_radius = 2  # Sample 2 pixels around center (5x5 area)
            black_threshold = 50
            
            # Process all coordinates in order
            for idx, (center_x, center_y) in enumerate(cell_coordinates):
                cx, cy = int(center_x), int(center_y)
                
                if 0 <= cy < img_array.shape[0] and 0 <= cx < img_array.shape[1]:
                    x1 = max(0, cx - sample_radius)
                    y1 = max(0, cy - sample_radius)
                    x2 = min(img_array.shape[1], cx + sample_radius + 1)
                    y2 = min(img_array.shape[0], cy + sample_radius + 1)
                    
                    sample_region = img_array[y1:y2, x1:x2]
                    
                    if sample_region.size > 0 and len(sample_region.shape) == 3 and sample_region.shape[2] == 3:
                        # RGB image - vectorized processing
                        pixels = sample_region.reshape(-1, 3)
                        # Filter out dark pixels (grid lines) and get median color
                        bright_mask = np.any(pixels > black_threshold, axis=1)
                        if np.any(bright_mask):
                            center_color = np.median(pixels[bright_mask], axis=0)
                        else:
                            center_color = np.median(pixels, axis=0)
                        
                        # Ensure 3-element RGB array
                        center_color = np.asarray(center_color).flatten()[:3]
                        if len(center_color) < 3:
                            center_color = np.pad(center_color, (0, 3 - len(center_color)), mode='edge')[:3]
                        
                        cell_colors[idx] = center_color.astype(int).tolist()
                    else:
                        # Fallback: use exact center pixel
                        center_color = img_array[cy, cx]
                        center_color = np.asarray(center_color).flatten()[:3]
                        if len(center_color) < 3:
                            center_color = np.pad(center_color, (0, 3 - len(center_color)), mode='edge')[:3]
                        cell_colors[idx] = center_color.astype(int).tolist()
                else:
                    # Invalid coordinates
                    cell_colors[idx] = [0, 0, 0]
            
            # Build simple cell data structure: list of dicts with row, col, coordinates, and color
            cell_data = []
            for idx, (center_x, center_y) in enumerate(cell_coordinates):
                row = idx // cols
                col = idx % cols
                color = cell_colors[idx]
                
                # Ensure color is a proper RGB array
                if isinstance(color, np.ndarray):
                    color_array = color.flatten()[:3]
                else:
                    color_array = np.array(color)[:3]
                
                # Convert to list of integers
                rgb_color = [int(c) for c in color_array[:3]]
                
                cell_data.append({
                    'row': row,
                    'col': col,
                    'center_x': int(center_x),
                    'center_y': int(center_y),
                    'color': rgb_color
                })
            
            return cell_data

        except Exception as e:
            print(f"Error getting cell data: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def _cluster_and_filter_lines(
        self,
        values: List[float],
        min_grid_size: int,
        max_grid_size: int
    ) -> List[float]:
        """
        Intelligently cluster and filter lines to find the main grid region.

        This method groups lines that are close together, identifies the most
        regular spacing pattern, and returns lines that form a valid grid.

        Parameters
        ----------
        values : List[float]
            List of line coordinates (y for horizontal, x for vertical).
        min_grid_size : int
            Minimum expected grid size.
        max_grid_size : int
            Maximum expected grid size.

        Returns
        -------
        List[float]
            Filtered list of line coordinates forming the main grid.
        """
        if not values or len(values) < 2:
            return values

        sorted_values = sorted(values)
        
        # First pass: filter very close lines (threshold=5 for tighter filtering)
        filtered = self._filter_close_values(sorted_values, threshold=5)
        
        if len(filtered) < 2:
            return filtered
        
        # Calculate spacing between consecutive lines
        spacings = []
        for i in range(len(filtered) - 1):
            spacings.append(filtered[i + 1] - filtered[i])
        
        if not spacings:
            return filtered
        
        # Find the most common spacing (mode of spacings, with tolerance)
        # This represents the cell size
        min_spacing = min(spacings)
        max_spacing = max(spacings)
        avg_spacing = sum(spacings) / len(spacings)
        
        # Reject if spacing is too small (less than 20 pixels indicates noise/UI elements)
        min_valid_spacing = 20
        if min_spacing < min_valid_spacing:
            # Filter out spacings that are too small
            valid_spacings = [s for s in spacings if s >= min_valid_spacing]
            if len(valid_spacings) < 2:
                # Not enough valid spacings, use fallback
                pass
            else:
                spacings = valid_spacings
                # Re-filter the lines to match valid spacings
                new_filtered = [filtered[0]]
                for i in range(1, len(filtered)):
                    spacing = filtered[i] - new_filtered[-1]
                    if spacing >= min_valid_spacing:
                        new_filtered.append(filtered[i])
                filtered = new_filtered
        
        if len(filtered) < 2:
            return filtered
        
        # Recalculate spacings after filtering
        spacings = [filtered[i + 1] - filtered[i] for i in range(len(filtered) - 1)]
        if not spacings:
            return filtered
        
        spacing_tolerance = min(spacings) * 0.2  # 20% tolerance
        spacing_groups = {}
        
        for spacing in spacings:
            # Find closest group
            matched = False
            for group_spacing in spacing_groups:
                if abs(spacing - group_spacing) <= spacing_tolerance:
                    spacing_groups[group_spacing].append(spacing)
                    matched = True
                    break
            if not matched:
                spacing_groups[spacing] = [spacing]
        
        # Get the most common spacing (largest group)
        if spacing_groups:
            dominant_spacing = max(spacing_groups.items(), key=lambda x: len(x[1]))[0]
            
            # Validate dominant spacing is reasonable (at least 20 pixels)
            if dominant_spacing < min_valid_spacing:
                # Spacing too small, likely not a real grid
                pass
            else:
                # Find the largest contiguous region with regular spacing
                # Expected grid size: (max_grid_size + 1) lines
                max_lines = max_grid_size + 1
                
                best_region = []
                best_score = 0
                
                # Try different starting positions
                for start_idx in range(len(filtered) - 1):
                    region = [filtered[start_idx]]
                    for i in range(start_idx + 1, len(filtered)):
                        expected_pos = region[-1] + dominant_spacing
                        actual_pos = filtered[i]
                        
                        # Check if this line fits the expected spacing
                        if abs(actual_pos - expected_pos) <= spacing_tolerance:
                            region.append(filtered[i])
                            if len(region) > max_lines:
                                break
                        elif actual_pos > expected_pos + spacing_tolerance:
                            # Gap too large, stop this region
                            break
                    
                    # Score based on length and regularity
                    if len(region) >= min_grid_size + 1:
                        # Calculate regularity score
                        if len(region) > 1:
                            region_spacings = [region[i+1] - region[i] for i in range(len(region)-1)]
                            avg_spacing = sum(region_spacings) / len(region_spacings)
                            variance = sum((s - avg_spacing)**2 for s in region_spacings) / len(region_spacings)
                            regularity_score = 1.0 / (1.0 + variance)  # Higher is better
                            score = len(region) * regularity_score
                            
                            if score > best_score:
                                best_score = score
                                best_region = region
                
                if best_region and len(best_region) >= min_grid_size + 1:
                    # Ensure we don't exceed max_grid_size
                    if len(best_region) > max_lines:
                        best_region = best_region[:max_lines]
                    return best_region
        
        # Fallback: if no good region found, use simple filtering with larger threshold
        # but limit to max_grid_size + 1 lines
        result = self._filter_close_values(sorted_values, threshold=10)
        if len(result) > max_grid_size + 1:
            # Take the middle section (most likely to be the main grid)
            excess = len(result) - (max_grid_size + 1)
            start_idx = excess // 2
            result = result[start_idx:start_idx + max_grid_size + 1]
        
        return result

    def _cluster_colors(
        self,
        cell_colors: List[np.ndarray],
        rows: int,
        cols: int,
        color_tolerance: float = 30.0
    ) -> List[List[int]]:
        """
        Cluster cell colors into regions.

        Parameters
        ----------
        cell_colors : List[np.ndarray]
            List of RGB color arrays for each cell.
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        color_tolerance : float, optional
            Color difference threshold for clustering. Default is 30.0.

        Returns
        -------
        List[List[int]]
            2D list of region IDs.
        """
        regions = [[0 for _ in range(cols)] for _ in range(rows)]
        region_id = 0
        unassigned = set(range(rows * cols))

        # Simple clustering: group cells with similar colors
        while unassigned:
            # Pick first unassigned cell
            cell_idx = min(unassigned)
            row = cell_idx // cols
            col = cell_idx % cols

            # Find all cells with similar color
            current_color = cell_colors[cell_idx]
            similar_cells = []

            for idx in unassigned:
                r = idx // cols
                c = idx % cols
                color = cell_colors[idx]
                # Calculate color distance
                color_diff = np.linalg.norm(current_color - color)
                if color_diff <= color_tolerance:
                    similar_cells.append(idx)

            # Assign region ID to similar cells
            for idx in similar_cells:
                r = idx // cols
                c = idx % cols
                regions[r][c] = region_id
                unassigned.remove(idx)

            region_id += 1

        return regions

    def _generate_region_colors(self, regions: List[List[int]]) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate distinct colors for each region for visualization.

        Parameters
        ----------
        regions : List[List[int]]
            2D list of region IDs.

        Returns
        -------
        Dict[int, Tuple[int, int, int]]
            Dictionary mapping region ID to RGB color.
        """
        unique_regions = set()
        for row in regions:
            unique_regions.update(row)

        colors = {}
        hue_step = 180 // max(len(unique_regions), 1)
        for i, region_id in enumerate(sorted(unique_regions)):
            hue = i * hue_step
            # Convert HSV to RGB
            hsv = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors[region_id] = tuple(int(c) for c in rgb)

        return colors

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
                    # Calculate if not in list
                    center_x = int(grid_x + (col + 0.5) * (grid_width / cols))
                    center_y = int(grid_y + (row + 0.5) * (grid_height / rows))
                    row_coords.append((center_x, center_y))
                idx += 1
            coords_2d.append(row_coords)

        return coords_2d

