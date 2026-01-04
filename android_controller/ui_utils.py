"""
UI text extraction utilities for Android UI Automator.

This module provides utilities for extracting text and coordinates from
UI hierarchy dumps obtained via UI Automator.
"""

import xml.etree.ElementTree as ET
import re
import tempfile
import os
import subprocess
from typing import Optional, Tuple, List, Dict, Any


def parse_bounds(bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse bounds string in format [x1,y1][x2,y2] to (x1, y1, x2, y2).

    Parameters
    ----------
    bounds_str : str
        Bounds string from UI Automator XML.

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Tuple of (x1, y1, x2, y2) or None if parsing fails.
    """
    if not bounds_str:
        return None
    # Match pattern [x1,y1][x2,y2]
    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if match:
        return tuple(map(int, match.groups()))
    return None


def get_center(bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate center point of bounds rectangle.

    Parameters
    ----------
    bounds : Tuple[int, int, int, int]
        Bounds as (x1, y1, x2, y2).

    Returns
    -------
    Tuple[int, int]
        Center point as (x, y).
    """
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def extract_text_elements_from_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Extract all text elements with coordinates from UI Automator XML dump.

    Parameters
    ----------
    xml_path : str
        Path to the UI Automator XML dump file.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing:
        - 'text': Text content
        - 'bounds': Tuple of (x1, y1, x2, y2)
        - 'center': Tuple of (x, y) center coordinates
        - 'source': 'text' or 'content-desc'
    """
    text_elements: List[Dict[str, Any]] = []
    seen_texts = set()  # To avoid duplicates

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for node in root.iter():
            # Get text attribute
            text = node.get('text', '').strip()
            bounds_str = node.get('bounds', '')
            bounds = parse_bounds(bounds_str)

            if text and bounds:
                # Create unique key to avoid exact duplicates
                text_key = f"{text}_{bounds}"
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    center = get_center(bounds)
                    text_elements.append({
                        'text': text,
                        'bounds': bounds,
                        'center': center,
                        'source': 'text'
                    })

            # Get content-desc attribute (often used for accessibility)
            content_desc = node.get('content-desc', '').strip()
            if content_desc and bounds:
                # Create unique key to avoid exact duplicates
                desc_key = f"{content_desc}_{bounds}"
                if desc_key not in seen_texts:
                    seen_texts.add(desc_key)
                    center = get_center(bounds)
                    text_elements.append({
                        'text': content_desc,
                        'bounds': bounds,
                        'center': center,
                        'source': 'content-desc'
                    })

    except ET.ParseError as e:
        raise ValueError(f"Error parsing UI dump XML: {e}")
    except Exception as e:
        raise ValueError(f"Error extracting text elements: {e}")

    return text_elements


def find_text_coordinates(
    vm_index: int,
    text_to_find: str,
    adb_path: str,
    adb_port: int,
    exact_match: bool = False,
    case_sensitive: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Find coordinates of a specific text element in the current UI.

    Parameters
    ----------
    vm_index : int
        Index of the BlueStacks instance.
    text_to_find : str
        Text to search for.
    adb_path : str
        Path to adb executable.
    adb_port : int
        ADB port number.
    exact_match : bool, optional
        If True, requires exact match. If False, matches if text contains the search string.
        Default is False.
    case_sensitive : bool, optional
        Whether the search should be case sensitive. Default is False.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing text element info with coordinates, or None if not found.
        Contains: 'text', 'bounds', 'center', 'source'
    """
    device_dump_path = "/sdcard/ui_dump.xml"
    local_dump_path = os.path.join(tempfile.gettempdir(), f"ui_dump_{vm_index}.xml")

    try:
        # Dump UI hierarchy to device
        dump_cmd = [adb_path, "-s", f"127.0.0.1:{adb_port}", "shell", "uiautomator", "dump", device_dump_path]
        subprocess.run(dump_cmd, capture_output=True, timeout=10)

        # Pull the dump file to local machine
        pull_cmd = [adb_path, "-s", f"127.0.0.1:{adb_port}", "pull", device_dump_path, local_dump_path]
        pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=10)

        if pull_result.returncode != 0:
            return None

        # Extract all text elements
        text_elements = extract_text_elements_from_xml(local_dump_path)

        # Search for matching text
        search_text = text_to_find if case_sensitive else text_to_find.lower()
        
        for elem in text_elements:
            elem_text = elem['text'] if case_sensitive else elem['text'].lower()
            
            if exact_match:
                if elem_text == search_text:
                    return elem
            else:
                if search_text in elem_text:
                    return elem

        return None

    except Exception as e:
        return None
    finally:
        # Clean up local dump file
        try:
            if os.path.exists(local_dump_path):
                os.remove(local_dump_path)
        except Exception:
            pass
        # Clean up device dump file
        try:
            subprocess.run(
                [adb_path, "-s", f"127.0.0.1:{adb_port}", "shell", "rm", device_dump_path],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass

