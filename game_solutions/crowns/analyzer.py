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
        is_horizontal: bool = True,
        estimated_grid_size: Optional[int] = None
    ) -> List[int]:
        """
        Detect grid lines using OpenCV-recommended morphological operations method.
        
        This method uses MORPH_OPEN to extract horizontal/vertical lines separately,
        which works robustly even with varying line thickness (thick region borders
        and thin internal grid lines).
        
        Parameters are dynamically adjusted based on grid size:
        - Small grids (5x5): thick lines = larger block size, thicker kernels, more iterations
        - Large grids (15x15): thin lines = smaller block size, thinner kernels, fewer iterations
        
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
        estimated_grid_size : Optional[int], optional
            Estimated grid size (e.g., 5, 9, 15). Used to adapt parameters.
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
            # Estimate from dimension (assume grid is between 3x3 and 15x15)
            # Use average cell size estimate
            possible_sizes = list(range(3, 16))
            avg_cell_size = np.mean([grid_dimension / size for size in possible_sizes])
            estimated_grid_size = int(round(grid_dimension / avg_cell_size))
            estimated_grid_size = max(3, min(15, estimated_grid_size))
        
        # Calculate adaptive parameters based on grid size
        # Small grids (5x5) have thick lines, large grids (15x15) have thin lines
        if estimated_grid_size <= 6:
            # Small grid: thick lines
            block_size = max(21, int(grid_region.shape[0] / estimated_grid_size * 0.3))
            if block_size % 2 == 0:
                block_size += 1  # Must be odd
            c_value = 5  # Higher C for better contrast
            kernel_thickness = 2  # Thicker kernel for thick lines
            morph_iterations = 3  # More iterations for thick lines
        elif estimated_grid_size <= 10:
            # Medium grid: medium lines
            block_size = max(15, int(grid_region.shape[0] / estimated_grid_size * 0.25))
            if block_size % 2 == 0:
                block_size += 1
            c_value = 3
            kernel_thickness = 1
            morph_iterations = 2
        else:
            # Large grid: thin lines
            block_size = max(9, int(grid_region.shape[0] / estimated_grid_size * 0.2))
            if block_size % 2 == 0:
                block_size += 1
            c_value = 2
            kernel_thickness = 1
            morph_iterations = 1
        
        # Step 1: Apply adaptive thresholding (handles varying lighting)
        # Block size adapts to grid size - larger for small grids, smaller for large grids
        binary = cv2.adaptiveThreshold(
            grid_region,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so dark lines become white
            block_size,
            c_value
        )
        
        # Step 2: Define structuring elements for morphological operations
        # The key is to use LONG kernels in the direction we want to extract
        # Kernel thickness adapts to grid size (thicker for small grids)
        # Calculate kernel length based on grid size
        # Should be long enough to span multiple cells but not too long
        if is_horizontal:
            # Horizontal kernel: wide enough to span most of the grid width
            kernel_length = int(grid_region.shape[1] * 0.6)  # 60% of width
            kernel_length = max(25, kernel_length)  # At least 25 pixels
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                (kernel_length, kernel_thickness)  # Wide, thickness adapts to grid size
            )
            
            # Use MORPH_OPEN to extract horizontal lines
            # MORPH_OPEN = erosion followed by dilation
            # This removes small objects and isolates horizontal lines
            detected_lines = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                horizontal_kernel,
                iterations=morph_iterations  # Adapts to grid size
            )
        else:
            # Vertical kernel: tall enough to span most of the grid height
            kernel_length = int(grid_region.shape[0] * 0.6)  # 60% of height
            kernel_length = max(25, kernel_length)  # At least 25 pixels
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_thickness, kernel_length)  # Tall, thickness adapts to grid size
            )
            
            # Use MORPH_OPEN to extract vertical lines
            detected_lines = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                vertical_kernel,
                iterations=morph_iterations  # Adapts to grid size
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
        # Use the estimated grid size to calculate cell size more accurately
        if is_horizontal:
            grid_dimension = grid_region.shape[0]
        else:
            grid_dimension = grid_region.shape[1]
        
        # Calculate cell size based on estimated grid size
        estimated_cell_size = int(grid_dimension / estimated_grid_size)
        
        # Window size for peak detection adapts to grid size
        # Small grids: larger window (thick lines), large grids: smaller window (thin lines)
        if estimated_grid_size <= 6:
            window_size = max(5, estimated_cell_size // 3)  # Larger window for thick lines
        elif estimated_grid_size <= 10:
            window_size = max(3, estimated_cell_size // 4)
        else:
            window_size = max(2, estimated_cell_size // 5)  # Smaller window for thin lines
        
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
        # Minimum spacing adapts to grid size
        if estimated_grid_size <= 6:
            min_spacing = max(5, estimated_cell_size // 3)  # Larger spacing for thick lines
        elif estimated_grid_size <= 10:
            min_spacing = max(3, estimated_cell_size // 4)
        else:
            min_spacing = max(2, estimated_cell_size // 5)  # Smaller spacing for thin lines
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
            
            # NEW APPROACH: Since grid boundaries are FIXED, try all possible grid sizes
            # and find which one best matches the detected line pattern
            # For each grid size N, we expect N+1 lines (N-1 internal + 2 borders)
            
            best_grid_size = None
            best_score = -1
            best_h_lines = None
            best_v_lines = None
            
            # Try each possible grid size
            for test_size in range(min_grid_size, max_grid_size + 1):
                # Detect lines with parameters optimized for this grid size
                h_lines = self._detect_grid_lines(grid_region, is_horizontal=True, estimated_grid_size=test_size)
                v_lines = self._detect_grid_lines(grid_region, is_horizontal=False, estimated_grid_size=test_size)
                
                if len(h_lines) < 2 or len(v_lines) < 2:
                    continue
                
                # Calculate expected spacing for this grid size
                expected_spacing_h = grid_height / test_size
                expected_spacing_v = grid_width / test_size
                
                # Score this grid size based on:
                # 1. Number of lines detected (should be close to test_size+1)
                # 2. Regularity of spacing (should match expected spacing)
                h_sorted = sorted(h_lines)
                v_sorted = sorted(v_lines)
                
                # Calculate actual spacings
                h_spacings = np.diff(h_sorted) if len(h_sorted) > 1 else []
                v_spacings = np.diff(v_sorted) if len(v_sorted) > 1 else []
                
                if len(h_spacings) == 0 or len(v_spacings) == 0:
                    continue
                
                # Score based on spacing regularity
                h_spacing_score = 1.0 / (1.0 + np.std(h_spacings) / np.mean(h_spacings)) if np.mean(h_spacings) > 0 else 0
                v_spacing_score = 1.0 / (1.0 + np.std(v_spacings) / np.mean(v_spacings)) if np.mean(v_spacings) > 0 else 0
                
                # Score based on how close spacing is to expected
                h_expected_score = 1.0 / (1.0 + abs(np.mean(h_spacings) - expected_spacing_h) / expected_spacing_h) if expected_spacing_h > 0 else 0
                v_expected_score = 1.0 / (1.0 + abs(np.mean(v_spacings) - expected_spacing_v) / expected_spacing_v) if expected_spacing_v > 0 else 0
                
                # Score based on number of lines (should be test_size+1, but allow some tolerance)
                expected_lines = test_size + 1
                h_line_count_score = 1.0 / (1.0 + abs(len(h_lines) - expected_lines) / expected_lines)
                v_line_count_score = 1.0 / (1.0 + abs(len(v_lines) - expected_lines) / expected_lines)
                
                # Combined score
                total_score = (h_spacing_score + v_spacing_score + h_expected_score + v_expected_score + 
                             h_line_count_score + v_line_count_score) / 6.0
                
                if self.debug:
                    print(f"Debug: Testing grid size {test_size}x{test_size}: "
                          f"h_lines={len(h_lines)}, v_lines={len(v_lines)}, "
                          f"score={total_score:.3f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_grid_size = test_size
                    best_h_lines = h_lines
                    best_v_lines = v_lines
            
            if best_grid_size is None:
                print("Error: Could not determine grid size from line detection.")
                return None
            
            if self.debug:
                print(f"Debug: Best grid size: {best_grid_size}x{best_grid_size} (score: {best_score:.3f})")
                print(f"Debug: Detected {len(best_h_lines)} horizontal lines, {len(best_v_lines)} vertical lines")
            
            # Use the best detected lines
            horizontal_lines_sorted = sorted(best_h_lines)
            vertical_lines_sorted = sorted(best_v_lines)
            
            # Calculate spacing between consecutive lines FIRST (before adding borders)
            if len(horizontal_lines_sorted) > 1:
                horizontal_spacings = [
                    horizontal_lines_sorted[i+1] - horizontal_lines_sorted[i]
                    for i in range(len(horizontal_lines_sorted) - 1)
                ]
                avg_h_spacing = np.mean(horizontal_spacings)
            else:
                avg_h_spacing = grid_height / best_grid_size if best_grid_size > 0 else grid_height / 15
            
            if len(vertical_lines_sorted) > 1:
                vertical_spacings = [
                    vertical_lines_sorted[i+1] - vertical_lines_sorted[i]
                    for i in range(len(vertical_lines_sorted) - 1)
                ]
                avg_v_spacing = np.mean(vertical_spacings)
            else:
                avg_v_spacing = grid_width / best_grid_size if best_grid_size > 0 else grid_width / 15
            
            # Add border lines based on POSITION, not count
            # Check if top border is missing (first line is not near 0)
            border_threshold = avg_h_spacing * 0.3
            if len(horizontal_lines_sorted) == 0 or horizontal_lines_sorted[0] > border_threshold:
                horizontal_lines_sorted.insert(0, 0)
            
            # Check if bottom border is missing (last line is not near grid_height)
            if len(horizontal_lines_sorted) == 0 or horizontal_lines_sorted[-1] < grid_height - border_threshold:
                horizontal_lines_sorted.append(grid_height - 1)
            
            # Check if left border is missing
            if len(vertical_lines_sorted) == 0 or vertical_lines_sorted[0] > border_threshold:
                vertical_lines_sorted.insert(0, 0)
            
            # Check if right border is missing
            if len(vertical_lines_sorted) == 0 or vertical_lines_sorted[-1] < grid_width - border_threshold:
                vertical_lines_sorted.append(grid_width - 1)
            
            # Recalculate grid size based on ACTUAL line count after adding borders
            # Grid size = number of lines - 1 (since NxN grid has N+1 lines)
            estimated_size_h = len(horizontal_lines_sorted) - 1
            estimated_size_v = len(vertical_lines_sorted) - 1
            
            # Use the average if they differ slightly, or the larger if they differ significantly
            if abs(estimated_size_h - estimated_size_v) <= 1:
                estimated_size = int(round((estimated_size_h + estimated_size_v) / 2))
            else:
                # If they differ significantly, use the one that matches the best_grid_size better
                # or use the larger one (more likely to be correct)
                estimated_size = max(estimated_size_h, estimated_size_v)
            
            # Ensure grid size is within valid range
            estimated_size = max(min_grid_size, min(max_grid_size, estimated_size))
            
            # Use the recalculated grid size
            estimated_rows = estimated_size
            estimated_cols = estimated_size
            
            if self.debug:
                print(f"Debug: Final grid size: {estimated_rows}x{estimated_cols}")
                print(f"Debug: Final line counts - H: {len(horizontal_lines_sorted)}, V: {len(vertical_lines_sorted)}")
                print(f"Debug: Average spacing - H: {avg_h_spacing:.1f}, V: {avg_v_spacing:.1f}")
            
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
                for line_y in horizontal_lines_sorted:
                    cv2.line(debug_img, (grid_x, grid_y + line_y), 
                           (grid_x + grid_width, grid_y + line_y), (255, 0, 0), 1)
                for line_x in vertical_lines_sorted:
                    cv2.line(debug_img, (grid_x + line_x, grid_y), 
                           (grid_x + line_x, grid_y + grid_height), (255, 0, 0), 1)
                
                # Draw cell centers
                for center_x, center_y in cell_coordinates:
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                
                cv2.imwrite('debug_grid.png', debug_img)
                print(f"Debug: Detected {len(horizontal_lines_sorted)} horizontal lines, {len(vertical_lines_sorted)} vertical lines")
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

