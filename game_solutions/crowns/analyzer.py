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

    def detect_grid(
        self,
        screenshot: Image.Image,
        min_grid_size: int = 3,
        max_grid_size: int = 15
    ) -> Optional[Dict[str, Any]]:
        """
        Detect grid boundaries and cell positions from a screenshot.
        
        New approach:
        1. Detect the grid as a large square/rectangle
        2. Count distinct color regions inside (excluding black grid lines)
        3. Determine grid size from region count
        4. Calculate cell positions

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
            # Grid is at (16, 166) to (884, 1035) - a 9x9 square grid
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
            
            # Step 2: Extract grid region and count distinct colors (excluding black)
            grid_region = img_cv[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width]
            
            # Convert to RGB for color analysis
            grid_rgb = cv2.cvtColor(grid_region, cv2.COLOR_BGR2RGB)
            
            # Reshape to list of pixels
            pixels = grid_rgb.reshape(-1, 3)
            
            # Filter out black/dark pixels (grid lines)
            # Grid lines are dark gray/blackish - use a higher threshold to exclude them
            # Region colors are bright, so we want pixels that are clearly not grid lines
            black_threshold = 50  # Increased threshold to better exclude dark grid lines
            non_black_mask = np.any(pixels > black_threshold, axis=1)
            non_black_pixels = pixels[non_black_mask]
            
            if len(non_black_pixels) == 0:
                return None
            
            # Cluster colors to find distinct regions
            # We expect exactly 9 distinct region colors for a 9x9 grid
            # Use k-means clustering to find the 9 most distinct colors
            from sklearn.cluster import KMeans
            
            # Try k-means with k=9 to find the 9 region colors
            target_colors = 9
            if len(non_black_pixels) < target_colors:
                # Not enough pixels, use simple quantization
                quantization_step = 20
                quantized = (non_black_pixels // quantization_step) * quantization_step
                quantized_tuples = [tuple(p) for p in quantized]
                unique_colors = set(quantized_tuples)
                best_k = len(unique_colors)
            else:
                try:
                    # Use k-means to find exactly 9 cluster centers
                    kmeans = KMeans(n_clusters=target_colors, random_state=42, n_init=10, max_iter=300)
                    kmeans.fit(non_black_pixels)
                    best_k = target_colors
                except Exception as e:
                    # Fallback to quantization if k-means fails
                    # Use adaptive quantization
                    quantization_step = 25  # Start with moderate quantization
                    quantized = (non_black_pixels // quantization_step) * quantization_step
                    quantized_tuples = [tuple(p) for p in quantized]
                    unique_colors = set(quantized_tuples)
                    best_k = len(unique_colors)
                    
                    # Adjust to get closer to 9 colors
                    if best_k < 7:
                        quantization_step = 15
                        quantized = (non_black_pixels // quantization_step) * quantization_step
                        quantized_tuples = [tuple(p) for p in quantized]
                        unique_colors = set(quantized_tuples)
                        best_k = len(unique_colors)
                    elif best_k > 12:
                        quantization_step = 35
                        quantized = (non_black_pixels // quantization_step) * quantization_step
                        quantized_tuples = [tuple(p) for p in quantized]
                        unique_colors = set(quantized_tuples)
                        best_k = len(unique_colors)
            
            # Step 3: Determine grid size from region count
            # For a square grid with N distinct region colors, the grid is typically N x N
            # User confirmed: 9 distinct colors = 9x9 grid
            
            estimated_rows = None
            estimated_cols = None
            
            # If we found exactly 9 colors, it's a 9x9 grid
            if best_k == 9:
                estimated_rows = 9
                estimated_cols = 9
            # The number of distinct colors typically equals the grid size for square grids
            elif min_grid_size <= best_k <= max_grid_size:
                estimated_rows = best_k
                estimated_cols = best_k
            else:
                # If outside range, default to 9x9 (most common) or closest valid size
                if 7 <= best_k <= 11:  # Close to 9
                    estimated_rows = 9
                    estimated_cols = 9
                else:
                    estimated_rows = max(min_grid_size, min(max_grid_size, best_k))
                    estimated_cols = estimated_rows
            
            if estimated_rows is None or estimated_cols is None:
                return None
            
            # Step 4: Calculate cell dimensions and positions
            cell_width = grid_width / estimated_cols
            cell_height = grid_height / estimated_rows
            
            # Validate cell dimensions
            min_cell_size = 20
            if cell_width < min_cell_size or cell_height < min_cell_size:
                return None
            
            # Calculate cell center coordinates
            cell_coordinates = []
            for row in range(estimated_rows):
                for col in range(estimated_cols):
                    center_x = grid_x + (col + 0.5) * cell_width
                    center_y = grid_y + (row + 0.5) * cell_height
                    cell_coordinates.append((int(center_x), int(center_y)))
            
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
                for center_x, center_y in cell_coordinates:
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.imwrite('debug_grid.png', debug_img)
            
            return result

        except Exception as e:
            print(f"Error detecting grid: {e}")
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

            # Sample the exact center color of each cell
            img_array = np.array(screenshot)
            cell_colors = []
            
            for idx, (center_x, center_y) in enumerate(cell_coordinates):
                # Sample the exact center pixel (or a very small area around center)
                # Use a small 3x3 or 5x5 area to avoid grid lines
                sample_radius = 2  # Sample 2 pixels around center (5x5 area)
                x1 = max(0, int(center_x) - sample_radius)
                y1 = max(0, int(center_y) - sample_radius)
                x2 = min(img_array.shape[1], int(center_x) + sample_radius + 1)
                y2 = min(img_array.shape[0], int(center_y) + sample_radius + 1)

                # Get the center pixel color directly
                if 0 <= int(center_y) < img_array.shape[0] and 0 <= int(center_x) < img_array.shape[1]:
                    # Sample a small area and get the median color (more robust than mean)
                    sample_region = img_array[y1:y2, x1:x2]
                    
                    if sample_region.size > 0:
                        # Check if sample_region is RGB (3 channels) or grayscale
                        if len(sample_region.shape) == 3 and sample_region.shape[2] == 3:
                            # RGB image - reshape to (N, 3)
                            pixels = sample_region.reshape(-1, 3)
                        elif len(sample_region.shape) == 2:
                            # Grayscale - convert to RGB by repeating
                            pixels = np.stack([sample_region.flatten(), sample_region.flatten(), sample_region.flatten()], axis=1)
                        else:
                            # Unexpected shape - use center pixel directly
                            center_color = img_array[int(center_y), int(center_x)]
                            if len(center_color.shape) == 0 or len(center_color) != 3:
                                # Fallback to RGB conversion
                                if len(center_color.shape) == 0:
                                    center_color = np.array([center_color, center_color, center_color])
                                else:
                                    center_color = np.array([center_color[0] if len(center_color) > 0 else 0,
                                                           center_color[1] if len(center_color) > 1 else 0,
                                                           center_color[2] if len(center_color) > 2 else 0])
                            cell_colors.append(center_color)
                            continue
                        
                        # Filter out very dark pixels (grid lines) and get median color
                        black_threshold = 50
                        bright_pixels = pixels[np.any(pixels > black_threshold, axis=1)]
                        
                        if len(bright_pixels) > 0:
                            # Use median for robustness against outliers
                            center_color = np.median(bright_pixels, axis=0)
                        else:
                            # If all pixels are dark, use median of all pixels
                            center_color = np.median(pixels, axis=0)
                        
                        # Ensure center_color is a 1D array with 3 elements (RGB)
                        center_color = np.asarray(center_color).flatten()
                        if len(center_color) != 3:
                            if len(center_color) == 1:
                                center_color = np.array([center_color[0], center_color[0], center_color[0]])
                            elif len(center_color) == 2:
                                center_color = np.array([center_color[0], center_color[1], center_color[1]])
                            else:
                                center_color = center_color[:3]
                        
                        cell_colors.append(center_color)
                    else:
                        # Fallback: use exact center pixel
                        center_color = img_array[int(center_y), int(center_x)]
                        # Ensure it's a 1D array with 3 elements
                        center_color = np.asarray(center_color).flatten()
                        if len(center_color) != 3:
                            if len(center_color) == 0:
                                center_color = np.array([0, 0, 0])
                            elif len(center_color) == 1:
                                center_color = np.array([center_color[0], center_color[0], center_color[0]])
                            elif len(center_color) == 2:
                                center_color = np.array([center_color[0], center_color[1], center_color[1]])
                            else:
                                center_color = center_color[:3]
                        cell_colors.append(center_color)
                else:
                    # Invalid coordinates
                    cell_colors.append([0, 0, 0])
            
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

