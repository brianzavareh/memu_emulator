"""
Coordinate conversion utilities for handling both relational and exact coordinates.

This module provides utilities for converting between relational (percentage-based)
and exact (pixel-based) coordinates, enabling modular game automation that works
with screenshots of different sizes.
"""

from typing import Union, Tuple, Optional
from PIL import Image


class CoordinateConverter:
    """
    Utility class for converting between relational and exact coordinates.
    
    Relational coordinates are expressed as floats between 0.0 and 1.0,
    representing percentages of the screen/screenshot dimensions.
    Exact coordinates are integer pixel positions.
    """

    @staticmethod
    def is_relational(coord: Union[int, float, Tuple[Union[int, float], Union[int, float]]]) -> bool:
        """
        Check if a coordinate is relational (float) or exact (int).
        
        Parameters
        ----------
        coord : Union[int, float, Tuple[Union[int, float], Union[int, float]]]
            Single coordinate value or tuple of (x, y) coordinates.
        
        Returns
        -------
        bool
            True if coordinate is relational (contains floats), False otherwise.
        """
        if isinstance(coord, tuple):
            return any(isinstance(c, float) for c in coord)
        return isinstance(coord, float)

    @staticmethod
    def to_exact(
        coord: Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]],
        width: int,
        height: int,
        coord_type: str = "xy"
    ) -> Union[int, Tuple[int, int]]:
        """
        Convert relational or exact coordinates to exact pixel coordinates.
        
        Parameters
        ----------
        coord : Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]]
            Coordinate(s) to convert. Can be:
            - (x, y) tuple with int or float values
            - Single int or float value
        width : int
            Screen/screenshot width in pixels.
        height : int
            Screen/screenshot height in pixels.
        coord_type : str, optional
            Type of coordinate: "xy" for (x, y), "x" for x only, "y" for y only.
            Default is "xy".
        
        Returns
        -------
        Union[int, Tuple[int, int]]
            Exact pixel coordinate(s) as int or (int, int) tuple.
        
        Examples
        --------
        >>> CoordinateConverter.to_exact((0.5, 0.5), 1920, 1080)
        (960, 540)
        >>> CoordinateConverter.to_exact((100, 200), 1920, 1080)
        (100, 200)
        >>> CoordinateConverter.to_exact(0.5, 1920, 1080, "x")
        960
        """
        if coord_type == "xy":
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise ValueError("For coord_type='xy', coord must be a tuple of (x, y)")
            
            x, y = coord
            if isinstance(x, float):
                x = int(x * width)
            if isinstance(y, float):
                y = int(y * height)
            
            return (int(x), int(y))
        
        elif coord_type == "x":
            if isinstance(coord, float):
                return int(coord * width)
            return int(coord)
        
        elif coord_type == "y":
            if isinstance(coord, float):
                return int(coord * height)
            return int(coord)
        
        else:
            raise ValueError(f"Invalid coord_type: {coord_type}. Must be 'xy', 'x', or 'y'")

    @staticmethod
    def to_relational(
        coord: Union[Tuple[int, int], int],
        width: int,
        height: int,
        coord_type: str = "xy"
    ) -> Union[float, Tuple[float, float]]:
        """
        Convert exact coordinates to relational (percentage-based) coordinates.
        
        Parameters
        ----------
        coord : Union[Tuple[int, int], int]
            Exact pixel coordinate(s) to convert.
        width : int
            Screen/screenshot width in pixels.
        height : int
            Screen/screenshot height in pixels.
        coord_type : str, optional
            Type of coordinate: "xy" for (x, y), "x" for x only, "y" for y only.
            Default is "xy".
        
        Returns
        -------
        Union[float, Tuple[float, float]]
            Relational coordinate(s) as float or (float, float) tuple.
        
        Examples
        --------
        >>> CoordinateConverter.to_relational((960, 540), 1920, 1080)
        (0.5, 0.5)
        >>> CoordinateConverter.to_relational(960, 1920, 1080, "x")
        0.5
        """
        if coord_type == "xy":
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise ValueError("For coord_type='xy', coord must be a tuple of (x, y)")
            
            x, y = coord
            return (x / width, y / height)
        
        elif coord_type == "x":
            return coord / width
        
        elif coord_type == "y":
            return coord / height
        
        else:
            raise ValueError(f"Invalid coord_type: {coord_type}. Must be 'xy', 'x', or 'y'")

    @staticmethod
    def convert_from_image(
        coord: Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]],
        image: Image.Image,
        coord_type: str = "xy"
    ) -> Union[int, Tuple[int, int]]:
        """
        Convert coordinates using an image's dimensions as reference.
        
        This is useful when working with screenshots - you can use
        relational coordinates based on the screenshot size, and they'll
        be converted to exact coordinates for the actual screen.
        
        Parameters
        ----------
        coord : Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]]
            Coordinate(s) to convert (relational or exact).
        image : Image.Image
            PIL Image object to use for dimension reference.
        coord_type : str, optional
            Type of coordinate: "xy" for (x, y), "x" for x only, "y" for y only.
            Default is "xy".
        
        Returns
        -------
        Union[int, Tuple[int, int]]
            Exact pixel coordinate(s).
        
        Examples
        --------
        >>> screenshot = Image.open("screenshot.png")
        >>> CoordinateConverter.convert_from_image((0.5, 0.5), screenshot)
        (960, 540)  # If screenshot is 1920x1080
        """
        width, height = image.size
        return CoordinateConverter.to_exact(coord, width, height, coord_type)

    @staticmethod
    def convert_from_screen_size(
        coord: Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]],
        screen_width: int,
        screen_height: int,
        coord_type: str = "xy"
    ) -> Union[int, Tuple[int, int]]:
        """
        Convert coordinates using explicit screen dimensions.
        
        Parameters
        ----------
        coord : Union[Tuple[Union[int, float], Union[int, float]], Union[int, float]]
            Coordinate(s) to convert (relational or exact).
        screen_width : int
            Screen width in pixels.
        screen_height : int
            Screen height in pixels.
        coord_type : str, optional
            Type of coordinate: "xy" for (x, y), "x" for x only, "y" for y only.
            Default is "xy".
        
        Returns
        -------
        Union[int, Tuple[int, int]]
            Exact pixel coordinate(s).
        """
        return CoordinateConverter.to_exact(coord, screen_width, screen_height, coord_type)

