"""
Configuration settings for AI Fitness Coach.

This module contains all configuration constants and settings used throughout
the application, including API keys, file paths, and application parameters.
"""

import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "ai_fitness" / "data"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"
TESTS_DIR = BASE_DIR / "tests"

# File paths
WORKOUT_LOG_FILE = DATA_DIR / "workout_log.csv"
APP_LOG_FILE = LOGS_DIR / "app.log"

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# AI Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.7
POSE_DETECTION_MODEL = "pose_landmarker.task"
EXERCISE_CLASSIFICATION_MODEL = "exercise_classifier.tflite"

# Streamlit settings
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"
STREAMLIT_DEBUG = False

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_fitness.db")
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API settings
API_TIMEOUT = 30
MAX_RETRIES = 3

class Settings:
    """Application settings class."""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.logs_dir = LOGS_DIR
        self.scripts_dir = SCRIPTS_DIR
        self.tests_dir = TESTS_DIR
        
        self.workout_log_file = WORKOUT_LOG_FILE
        self.app_log_file = APP_LOG_FILE
        
        self.camera_index = CAMERA_INDEX
        self.camera_width = CAMERA_WIDTH
        self.camera_height = CAMERA_HEIGHT
        self.camera_fps = CAMERA_FPS
        
        self.model_confidence_threshold = MODEL_CONFIDENCE_THRESHOLD
        self.pose_detection_model = POSE_DETECTION_MODEL
        self.exercise_classification_model = EXERCISE_CLASSIFICATION_MODEL
        
        self.streamlit_port = STREAMLIT_PORT
        self.streamlit_host = STREAMLIT_HOST
        self.streamlit_debug = STREAMLIT_DEBUG
        
        self.database_url = DATABASE_URL
        self.database_echo = DATABASE_ECHO
        
        self.log_level = LOG_LEVEL
        self.log_format = LOG_FORMAT
        
        self.api_timeout = API_TIMEOUT
        self.max_retries = MAX_RETRIES

def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()

# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()
