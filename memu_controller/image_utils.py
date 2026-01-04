"""
Image processing utilities for screenshot analysis and game automation.

This module provides utilities for processing screenshots, finding elements,
and performing image-based operations for game automation.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import io
import os
import cv2


class ImageProcessor:
    """
    Image processing utilities for screenshot analysis.

    This class provides methods for loading, processing, and analyzing
    screenshots from the emulator for game automation purposes.
    """

    @staticmethod
    def load_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
        """
        Load an image from bytes.

        Parameters
        ----------
        image_bytes : bytes
            Image data as bytes.

        Returns
        -------
        Optional[Image.Image]
            PIL Image object, or None if loading failed.
        """
        # Try multiple loading methods
        img = None
        method1_error = None
        method2_error = None
        method3_error = None
        
        try:
            # Method 1: Direct BytesIO
            try:
                img = Image.open(io.BytesIO(image_bytes))
            except Exception as e1:
                method1_error = str(e1)
                # Method 2: Try OpenCV (more tolerant of PNG corruption)
                try:
                    import numpy as np
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_cv is not None:
                        # Convert BGR to RGB for PIL
                        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_cv_rgb)
                    else:
                        raise Exception("OpenCV failed to decode image")
                except Exception as e2:
                    method2_error = str(e2)
                    # Method 3: Write to temp file and read (sometimes works better with corrupted PNGs)
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            tmp_path = tmp_file.name
                            tmp_file.write(image_bytes)
                        img = Image.open(tmp_path)
                        os.unlink(tmp_path)
                    except Exception as e3:
                        method3_error = str(e3)
                        # All methods failed - return None
                        return None
            
            # If we got here, img was successfully loaded by one of the methods
            if img is None:
                return None
            return img
        except Exception as e:
            print(f"Error loading image from bytes: {e}")
            return None

    @staticmethod
    def load_from_file(file_path: str) -> Optional[Image.Image]:
        """
        Load an image from file.

        Parameters
        ----------
        file_path : str
            Path to the image file.

        Returns
        -------
        Optional[Image.Image]
            PIL Image object, or None if loading failed.
        """
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"Error loading image from file: {e}")
            return None

    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format (numpy array).

        Parameters
        ----------
        pil_image : Image.Image
            PIL Image object.

        Returns
        -------
        np.ndarray
            OpenCV image array (BGR format).
        """
        # Convert PIL RGB to OpenCV BGR
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL Image.

        Parameters
        ----------
        cv2_image : np.ndarray
            OpenCV image array (BGR format).

        Returns
        -------
        Image.Image
            PIL Image object (RGB format).
        """
        # Convert OpenCV BGR to PIL RGB
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def find_template(
        screenshot: Image.Image,
        template_path: str,
        threshold: float = 0.8
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a template image within a screenshot using template matching.

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot image to search in.
        template_path : str
            Path to the template image file.
        threshold : float, optional
            Matching threshold (0.0 to 1.0). Default is 0.8.

        Returns
        -------
        Optional[Tuple[int, int, int, int]]
            Tuple of (x, y, width, height) of found template, or None if not found.
        """
        try:
            # Load template
            template = Image.open(template_path)
            
            # Convert to OpenCV format
            screenshot_cv = ImageProcessor.pil_to_cv2(screenshot)
            template_cv = ImageProcessor.pil_to_cv2(template)
            
            # Perform template matching
            result = cv2.matchTemplate(screenshot_cv, template_cv, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                h, w = template_cv.shape[:2]
                x, y = max_loc
                return (x, y, w, h)
            
            return None
        except Exception as e:
            print(f"Error finding template: {e}")
            return None

    @staticmethod
    def find_color_region(
        screenshot: Image.Image,
        color: Tuple[int, int, int],
        tolerance: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find regions in the screenshot matching a specific color.

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot to analyze.
        color : Tuple[int, int, int]
            RGB color to find.
        tolerance : int, optional
            Color matching tolerance. Default is 10.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            List of (x, y, width, height) bounding boxes of matching regions.
        """
        try:
            # Convert to numpy array
            img_array = np.array(screenshot)
            
            # Create color mask
            lower = np.array([max(0, c - tolerance) for c in color])
            upper = np.array([min(255, c + tolerance) for c in color])
            
            # Convert to OpenCV format and find contours
            img_cv = ImageProcessor.pil_to_cv2(screenshot)
            mask = cv2.inRange(img_cv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get bounding boxes
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Filter small regions
                    regions.append((x, y, w, h))
            
            return regions
        except Exception as e:
            print(f"Error finding color region: {e}")
            return []

    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get the center coordinates of a bounding box.

        Parameters
        ----------
        bbox : Tuple[int, int, int, int]
            Bounding box as (x, y, width, height).

        Returns
        -------
        Tuple[int, int]
            Center coordinates as (x, y).
        """
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    @staticmethod
    def crop_region(
        screenshot: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[Image.Image]:
        """
        Crop a region from the screenshot.

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot image.
        bbox : Tuple[int, int, int, int]
            Bounding box as (x, y, width, height).

        Returns
        -------
        Optional[Image.Image]
            Cropped image, or None if failed.
        """
        try:
            x, y, w, h = bbox
            return screenshot.crop((x, y, x + w, y + h))
        except Exception as e:
            print(f"Error cropping region: {e}")
            return None

    @staticmethod
    def save_image(image: Image.Image, file_path: str) -> bool:
        """
        Save an image to file.

        Parameters
        ----------
        image : Image.Image
            Image to save.
        file_path : str
            Path to save the image.

        Returns
        -------
        bool
            True if saved successfully, False otherwise.
        """
        try:
            image.save(file_path)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    @staticmethod
    def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """
        Resize an image.

        Parameters
        ----------
        image : Image.Image
            Image to resize.
        width : int
            Target width.
        height : int
            Target height.

        Returns
        -------
        Image.Image
            Resized image.
        """
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def find_text_region(
        screenshot: Image.Image,
        text_color: Optional[Tuple[int, int, int]] = None,
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find text-like regions in the screenshot based on color contrast.

        Parameters
        ----------
        screenshot : Image.Image
            Screenshot to analyze.
        text_color : Optional[Tuple[int, int, int]], optional
            Expected text color (RGB). If None, uses automatic detection.
        background_color : Optional[Tuple[int, int, int]], optional
            Expected background color (RGB). If None, uses automatic detection.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            List of bounding boxes for potential text regions.
        """
        try:
            img_cv = ImageProcessor.pil_to_cv2(screenshot)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find text-like regions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter for text-like aspect ratios
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 10 and w > 20 and h > 10:
                    regions.append((x, y, w, h))
            
            return regions
        except Exception as e:
            print(f"Error finding text regions: {e}")
            return []

