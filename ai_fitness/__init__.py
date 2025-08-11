"""
AI Fitness Coach - Main Package

Package metadata and lightweight initialization only. Heavy submodules are not
imported at package import time to remain compatible with environments where
optional native dependencies (e.g., OpenCV) may be unavailable.
"""

__version__ = "1.0.0"
__author__ = "AI Fitness Team"

# Intentionally avoid importing heavy submodules here (like OpenCV/MediaPipe users)
# to keep top-level import safe on constrained platforms (e.g., Streamlit Cloud).

__all__ = [
    # Expose version/author only by default; import submodules explicitly where needed
]
