"""
Image processing and OCR utilities for the Interactive Screenshot Processor.

This module provides utilities for OpenCV filtering, OCR operations, and
coordinate conversion for the screenshot processor UI.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import cv2

# Try to import EasyOCR
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# Try to import pytesseract
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

from android_controller.image_utils import ImageProcessor


class FilterProcessor:
    """
    OpenCV filter processing utilities.
    
    Provides methods to apply various OpenCV filters to images with
    configurable parameters.
    """
    
    @staticmethod
    def apply_threshold(
        image: np.ndarray,
        filter_type: str = "binary",
        value: int = 127,
        max_value: int = 255,
        adaptive_method: str = "mean",
        block_size: int = 11,
        c_value: float = 2.0
    ) -> np.ndarray:
        """
        Apply threshold filter to image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale).
        filter_type : str, optional
            Type of threshold: "binary", "binary_inv", "adaptive", "otsu".
            Default is "binary".
        value : int, optional
            Threshold value for binary operations. Default is 127.
        max_value : int, optional
            Maximum value for binary operations. Default is 255.
        adaptive_method : str, optional
            Adaptive method: "mean" or "gaussian". Default is "mean".
        block_size : int, optional
            Block size for adaptive threshold. Default is 11.
        c_value : float, optional
            C value for adaptive threshold. Default is 2.0.
        
        Returns
        -------
        np.ndarray
            Thresholded image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if filter_type == "binary":
            _, result = cv2.threshold(gray, value, max_value, cv2.THRESH_BINARY)
        elif filter_type == "binary_inv":
            _, result = cv2.threshold(gray, value, max_value, cv2.THRESH_BINARY_INV)
        elif filter_type == "adaptive":
            method = cv2.ADAPTIVE_THRESH_MEAN_C if adaptive_method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if block_size % 2 == 0:
                block_size += 1
            result = cv2.adaptiveThreshold(gray, max_value, method, cv2.THRESH_BINARY, block_size, c_value)
        elif filter_type == "otsu":
            _, result = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            result = gray
        
        return result
    
    @staticmethod
    def apply_blur(
        image: np.ndarray,
        blur_type: str = "gaussian",
        kernel_size: int = 5,
        sigma_x: float = 1.0,
        sigma_y: float = 0.0
    ) -> np.ndarray:
        """
        Apply blur filter to image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
        blur_type : str, optional
            Type of blur: "gaussian", "median", "bilateral". Default is "gaussian".
        kernel_size : int, optional
            Kernel size. Default is 5.
        sigma_x : float, optional
            Gaussian sigma X. Default is 1.0.
        sigma_y : float, optional
            Gaussian sigma Y. Default is 0.0.
        
        Returns
        -------
        np.ndarray
            Blurred image.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if blur_type == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigma_y)
        elif blur_type == "median":
            return cv2.medianBlur(image, kernel_size)
        elif blur_type == "bilateral":
            return cv2.bilateralFilter(image, kernel_size, sigma_x * 80, sigma_y * 80)
        else:
            return image
    
    @staticmethod
    def apply_morphology(
        image: np.ndarray,
        operation: str = "open",
        kernel_size: int = 3,
        kernel_shape: str = "rect",
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations to image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (binary/grayscale).
        operation : str, optional
            Operation: "erode", "dilate", "open", "close", "gradient", "tophat", "blackhat".
            Default is "open".
        kernel_size : int, optional
            Kernel size. Default is 3.
        kernel_shape : str, optional
            Kernel shape: "rect", "ellipse", "cross". Default is "rect".
        iterations : int, optional
            Number of iterations. Default is 1.
        
        Returns
        -------
        np.ndarray
            Processed image.
        """
        shape_map = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS
        }
        kernel = cv2.getStructuringElement(shape_map.get(kernel_shape, cv2.MORPH_RECT), (kernel_size, kernel_size))
        
        op_map = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT
        }
        
        if operation in op_map:
            return cv2.morphologyEx(image, op_map[operation], kernel, iterations=iterations)
        else:
            return image
    
    @staticmethod
    def apply_edge_detection(
        image: np.ndarray,
        edge_type: str = "canny",
        threshold1: int = 50,
        threshold2: int = 150,
        sobel_kernel: int = 3
    ) -> np.ndarray:
        """
        Apply edge detection to image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale).
        edge_type : str, optional
            Edge detection type: "canny", "sobel_x", "sobel_y", "sobel_xy", "laplacian".
            Default is "canny".
        threshold1 : int, optional
            Canny threshold 1. Default is 50.
        threshold2 : int, optional
            Canny threshold 2. Default is 150.
        sobel_kernel : int, optional
            Sobel kernel size. Default is 3.
        
        Returns
        -------
        np.ndarray
            Edge detected image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if edge_type == "canny":
            return cv2.Canny(gray, threshold1, threshold2)
        elif edge_type == "sobel_x":
            return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif edge_type == "sobel_y":
            return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        elif edge_type == "sobel_xy":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            return np.sqrt(sobelx**2 + sobely**2)
        elif edge_type == "laplacian":
            return cv2.Laplacian(gray, cv2.CV_64F, ksize=sobel_kernel)
        else:
            return gray
    
    @staticmethod
    def apply_color_space(
        image: np.ndarray,
        color_space: str = "grayscale"
    ) -> np.ndarray:
        """
        Convert image to different color space.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (BGR format).
        color_space : str, optional
            Color space: "grayscale", "hsv", "lab", "rgb". Default is "grayscale".
        
        Returns
        -------
        np.ndarray
            Converted image.
        """
        if color_space == "grayscale":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        elif color_space == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == "lab":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image
    
    @staticmethod
    def apply_contour_detection(
        image: np.ndarray,
        min_area: int = 100,
        max_area: int = 1000000,
        mode: str = "external",
        method: str = "simple"
    ) -> Tuple[np.ndarray, List]:
        """
        Detect contours in image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (binary/grayscale).
        min_area : int, optional
            Minimum contour area. Default is 100.
        max_area : int, optional
            Maximum contour area. Default is 1000000.
        mode : str, optional
            Contour retrieval mode: "external", "list", "tree", "ccomp".
            Default is "external".
        method : str, optional
            Contour approximation method: "simple", "none". Default is "simple".
        
        Returns
        -------
        Tuple[np.ndarray, List]
            Tuple of (image with contours drawn, list of contours).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure binary
        if len(np.unique(gray)) > 2:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = gray
        
        mode_map = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "tree": cv2.RETR_TREE,
            "ccomp": cv2.RETR_CCOMP
        }
        
        method_map = {
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "none": cv2.CHAIN_APPROX_NONE
        }
        
        contours, hierarchy = cv2.findContours(
            binary,
            mode_map.get(mode, cv2.RETR_EXTERNAL),
            method_map.get(method, cv2.CHAIN_APPROX_SIMPLE)
        )
        
        # Filter by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered_contours.append(contour)
        
        # Draw contours
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
        
        return result, filtered_contours
    
    @staticmethod
    def apply_hough_lines(
        image: np.ndarray,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10
    ) -> Tuple[np.ndarray, List]:
        """
        Detect lines using Hough transform.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (binary/grayscale).
        rho : float, optional
            Distance resolution. Default is 1.0.
        theta : float, optional
            Angle resolution. Default is pi/180.
        threshold : int, optional
            Accumulator threshold. Default is 100.
        min_line_length : int, optional
            Minimum line length. Default is 50.
        max_line_gap : int, optional
            Maximum line gap. Default is 10.
        
        Returns
        -------
        Tuple[np.ndarray, List]
            Tuple of (image with lines drawn, list of lines).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        lines = cv2.HoughLinesP(
            gray,
            rho,
            theta,
            threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        line_list = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                line_list.append((x1, y1, x2, y2))
        
        return result, line_list
    
    @staticmethod
    def apply_hough_circles(
        image: np.ndarray,
        method: str = "hough",
        dp: float = 1.0,
        min_dist: float = 50.0,
        param1: float = 50.0,
        param2: float = 30.0,
        min_radius: int = 0,
        max_radius: int = 0
    ) -> Tuple[np.ndarray, List]:
        """
        Detect circles using Hough transform.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale).
        method : str, optional
            Detection method. Default is "hough".
        dp : float, optional
            Inverse ratio of accumulator resolution. Default is 1.0.
        min_dist : float, optional
            Minimum distance between centers. Default is 50.0.
        param1 : float, optional
            Upper threshold for edge detection. Default is 50.0.
        param2 : float, optional
            Accumulator threshold. Default is 30.0.
        min_radius : int, optional
            Minimum circle radius. Default is 0.
        max_radius : int, optional
            Maximum circle radius. Default is 0.
        
        Returns
        -------
        Tuple[np.ndarray, List]
            Tuple of (image with circles drawn, list of circles).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        circle_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(result, center, radius, (255, 0, 0), 2)
                cv2.circle(result, center, 2, (0, 255, 0), 3)
                circle_list.append((center[0], center[1], radius))
        
        return result, circle_list


class OCRAnalyzer:
    """
    OCR and analysis operations for screenshot processing.
    """
    
    def __init__(self):
        """Initialize OCR analyzer."""
        self.easyocr_reader = None
        if HAS_EASYOCR:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception:
                self.easyocr_reader = None
    
    def detect_digits(
        self,
        image: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect digits in image using OCR.
        
        Parameters
        ----------
        image : Image.Image
            Input image.
        region : Optional[Tuple[int, int, int, int]], optional
            Region to analyze (x, y, width, height). If None, analyzes full image.
            Default is None.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of detected digits with positions and values.
        """
        if region:
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
        else:
            cropped = image
        
        results = []
        
        # Try EasyOCR first
        if self.easyocr_reader:
            try:
                img_array = np.array(cropped)
                ocr_results = self.easyocr_reader.readtext(img_array)
                for result in ocr_results:
                    bbox, text, confidence = result
                    # Filter for digits only
                    digit_text = ''.join(c for c in text if c.isdigit())
                    if digit_text:
                        # Calculate center of bbox
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        center_x = int(sum(x_coords) / len(x_coords))
                        center_y = int(sum(y_coords) / len(y_coords))
                        
                        if region:
                            center_x += x
                            center_y += y
                        
                        results.append({
                            "text": digit_text,
                            "confidence": float(confidence),
                            "position": (center_x, center_y),
                            "bbox": bbox
                        })
            except Exception as e:
                pass
        
        # Fallback to Tesseract
        if not results and HAS_TESSERACT:
            try:
                img_array = np.array(cropped)
                data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)
                for i, text in enumerate(data['text']):
                    if text.strip() and text.isdigit():
                        x_pos = data['left'][i] + (region[0] if region else 0)
                        y_pos = data['top'][i] + (region[1] if region else 0)
                        w_pos = data['width'][i]
                        h_pos = data['height'][i]
                        conf = float(data['conf'][i])
                        
                        results.append({
                            "text": text,
                            "confidence": conf / 100.0,
                            "position": (x_pos + w_pos // 2, y_pos + h_pos // 2),
                            "bbox": [(x_pos, y_pos), (x_pos + w_pos, y_pos), 
                                    (x_pos + w_pos, y_pos + h_pos), (x_pos, y_pos + h_pos)]
                        })
            except Exception:
                pass
        
        return results
    
    def detect_grid(
        self,
        image: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Detect grid structure in image.
        
        Parameters
        ----------
        image : Image.Image
            Input image.
        region : Optional[Tuple[int, int, int, int]], optional
            Region to analyze (x, y, width, height). If None, analyzes full image.
            Default is None.
        
        Returns
        -------
        Dict[str, Any]
            Grid detection results with lines and cells.
        """
        if region:
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
        else:
            cropped = image
        
        img_cv = ImageProcessor.pil_to_cv2(cropped)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Find line positions
        h_projection = np.sum(horizontal_lines, axis=1)
        v_projection = np.sum(vertical_lines, axis=0)
        
        h_threshold = np.max(h_projection) * 0.5
        v_threshold = np.max(v_projection) * 0.5
        
        h_lines = [i for i, val in enumerate(h_projection) if val > h_threshold]
        v_lines = [i for i, val in enumerate(v_projection) if val > v_threshold]
        
        # Estimate grid size
        rows = len(h_lines) - 1 if len(h_lines) > 1 else 0
        cols = len(v_lines) - 1 if len(v_lines) > 1 else 0
        
        # Adjust coordinates if region was specified
        if region:
            h_lines = [line + y for line in h_lines]
            v_lines = [line + x for line in v_lines]
        
        return {
            "horizontal_lines": h_lines,
            "vertical_lines": v_lines,
            "rows": rows,
            "cols": cols,
            "cells": rows * cols if rows > 0 and cols > 0 else 0
        }
    
    def detect_color(
        self,
        image: Image.Image,
        position: Optional[Tuple[int, int]] = None,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Detect color at position or average color in region.
        
        Parameters
        ----------
        image : Image.Image
            Input image.
        position : Optional[Tuple[int, int]], optional
            Position to sample (x, y). Default is None.
        region : Optional[Tuple[int, int, int, int]], optional
            Region to analyze (x, y, width, height). Default is None.
        
        Returns
        -------
        Dict[str, Any]
            Color information (RGB values).
        """
        if position:
            x, y = position
            if 0 <= x < image.width and 0 <= y < image.height:
                rgb = image.getpixel((x, y))
                return {
                    "type": "point",
                    "position": position,
                    "rgb": rgb,
                    "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                }
        elif region:
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            img_array = np.array(cropped)
            avg_rgb = tuple(int(val) for val in img_array.mean(axis=(0, 1)))
            return {
                "type": "region_average",
                "region": region,
                "rgb": avg_rgb,
                "hex": f"#{avg_rgb[0]:02x}{avg_rgb[1]:02x}{avg_rgb[2]:02x}"
            }
        
        return {"error": "No position or region specified"}


class CoordinateConverter:
    """
    Utility for converting between absolute and percentage-based coordinates.
    """
    
    @staticmethod
    def absolute_to_percentage(
        x: int,
        y: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert absolute coordinates to percentage-based.
        
        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        width : int
            Width.
        height : int
            Height.
        image_width : int
            Image width.
        image_height : int
            Image height.
        
        Returns
        -------
        Tuple[float, float, float, float]
            Percentage coordinates (x_pct, y_pct, width_pct, height_pct).
        """
        x_pct = x / image_width if image_width > 0 else 0.0
        y_pct = y / image_height if image_height > 0 else 0.0
        width_pct = width / image_width if image_width > 0 else 0.0
        height_pct = height / image_height if image_height > 0 else 0.0
        
        return (x_pct, y_pct, width_pct, height_pct)
    
    @staticmethod
    def percentage_to_absolute(
        x_pct: float,
        y_pct: float,
        width_pct: float,
        height_pct: float,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert percentage-based coordinates to absolute.
        
        Parameters
        ----------
        x_pct : float
            X percentage.
        y_pct : float
            Y percentage.
        width_pct : float
            Width percentage.
        height_pct : float
            Height percentage.
        image_width : int
            Image width.
        image_height : int
            Image height.
        
        Returns
        -------
        Tuple[int, int, int, int]
            Absolute coordinates (x, y, width, height).
        """
        x = int(x_pct * image_width)
        y = int(y_pct * image_height)
        width = int(width_pct * image_width)
        height = int(height_pct * image_height)
        
        return (x, y, width, height)

