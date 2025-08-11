"""
Utility functions and helper modules for AI Fitness Coach.

Contains common utilities, file operations, logging, and other
helper functions used throughout the application.
"""

from .file_utils import *
from .logger import *

__all__ = [
    "ensure_directory",
    "get_file_size",
    "get_logger",
    "setup_logging"
]
