"""
Data access layer package for AI Fitness Coach.

Contains database connections, data models, and data access objects
for managing workout and fitness data.
"""

from .database import DatabaseManager

__all__ = ["DatabaseManager"]
