"""
AI Fitness Coach - Main Package

A comprehensive AI-powered fitness coaching application that provides
personalized workout analysis, real-time form correction, and fitness tracking.
"""

__version__ = "1.0.0"
__author__ = "AI Fitness Team"

from .core.ai_analyzer import AIAnalyzer
from .core.workout_analytics import WorkoutAnalytics
from .ui.streamlit_app import main

__all__ = [
    "AIAnalyzer",
    "WorkoutAnalytics", 
    "main"
]
