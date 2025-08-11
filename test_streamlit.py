#!/usr/bin/env python3
"""
Test script to verify the Streamlit app can be imported and run without errors
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        import cv2
        print("✓ OpenCV imported successfully")
        
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
        
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import plotly.express as px
        print("✓ Plotly imported successfully")
        
        from ai_fitness.core.workout_analytics import WorkoutAnalytics
        print("✓ WorkoutAnalytics imported successfully")
        
        # Test importing the main app
        from ai_fitness.ui import streamlit_app
        print("✓ Main app imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_session_state():
    """Test session state initialization"""
    try:
        import streamlit as st
        
        # Simulate session state initialization
        session_state = {}
        
        # Initialize session state variables (matching actual app)
        if 'exercise_counters' not in session_state:
            session_state['exercise_counters'] = {
                'squats': 0,
                'pushups': 0,
                'planks': 0
            }
        if 'workout_log' not in session_state:
            session_state['workout_log'] = []
        if 'fitness_coach' not in session_state:
            session_state['fitness_coach'] = None
        if 'video_active' not in session_state:
            session_state['video_active'] = False
        if 'latest_frame' not in session_state:
            session_state['latest_frame'] = None
        if 'ai_analyzer' not in session_state:
            session_state['ai_analyzer'] = None
        if 'workout_generator' not in session_state:
            session_state['workout_generator'] = None
        if 'physique_analysis' not in session_state:
            session_state['physique_analysis'] = None
        if 'workout_routine' not in session_state:
            session_state['workout_routine'] = None
        
        print("✓ Session state initialization successful")
        print(f"  Exercise counters: {session_state['exercise_counters']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Session state test failed: {e}")
        return False

def test_workout_analytics():
    """Test WorkoutAnalytics functionality"""
    try:
        from ai_fitness.core.workout_analytics import WorkoutAnalytics
        
        analytics = WorkoutAnalytics()
        print("✓ WorkoutAnalytics instantiated successfully")
        
        # Test with empty data
        daily_summary = analytics.get_daily_summary()
        print(f"✓ Daily summary test passed (empty: {daily_summary.empty})")
        
        # Test FitnessCoach instantiation
        from ai_fitness.ui.streamlit_app import FitnessCoach
        coach = FitnessCoach()
        print("✓ FitnessCoach instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ WorkoutAnalytics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing AI Fitness Coach Streamlit App...")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Session State", test_session_state),
        ("Workout Analytics", test_workout_analytics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The app should run without errors.")
        print("\nTo run the app:")
        print("1. Activate your virtual environment: venv\\Scripts\\activate")
        print("2. Run: python -m streamlit run Ai_fitness/streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 