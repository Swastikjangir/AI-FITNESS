"""
Core business logic package for AI Fitness Coach.

Lightweight package initialization: we intentionally avoid importing heavy
submodules here (e.g., ones that depend on OpenCV/MediaPipe) to keep
`import ai_fitness.core` safe in constrained environments (such as
Streamlit Cloud). Import submodules explicitly where needed, e.g.:

    from ai_fitness.core.workout_analytics import WorkoutAnalytics
    from ai_fitness.core.ai_analyzer import AIAnalyzer
    from ai_fitness.core.data_processing import DataProcessor
"""

__all__ = [
    "AIAnalyzer",
    "WorkoutAnalytics",
    "DataProcessor",
]
