"""
Core business logic package for AI Fitness Coach.

Contains the main business logic components including AI analysis,
workout analytics, and data processing.
"""

from .ai_analyzer import AIAnalyzer
from .workout_analytics import WorkoutAnalytics
from .data_processing import DataProcessor

__all__ = [
    "AIAnalyzer",
    "WorkoutAnalytics",
    "DataProcessor"
]
