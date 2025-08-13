#!/usr/bin/env python3
"""
AI Fitness Coach - Main Application Entry Point

This script serves as the main entry point for running the AI Fitness Coach application.
It can be used to run the Streamlit UI, run tests, or perform other application tasks.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_streamlit():
    """Run the Streamlit application"""
    try:
        import subprocess
        import sys
        
        print("Starting AI Fitness Coach Streamlit Application...")
        print("The application will open in your default web browser.")
        print("If it doesn't open automatically, navigate to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the application.")
        
        # Run the Streamlit app using subprocess
        streamlit_path = os.path.join(project_root, "ai_fitness", "ui", "streamlit_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path])
        
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Please install the required dependencies using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting Streamlit application: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite"""
    try:
        import unittest
        
        print("Running AI Fitness Coach Test Suite...")
        
        # Discover and run tests
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern='test_*.py')
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        if result.wasSuccessful():
            print("\n✅ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

def run_ai_analysis():
    """Run AI analysis on camera input"""
    try:
        from ai_fitness.core.ai_analyzer import AIAnalyzer
        from ai_fitness.services.camera_service import CameraService
        import cv2
        
        print("Starting AI Analysis...")
        print("Press 'q' to quit, 'f' for fast mode, 'b' for balanced mode, 'q' for quality mode")
        
        # Initialize AI analyzer with balanced mode
        ai_analyzer = AIAnalyzer(performance_mode="balanced")
        
        # Initialize camera service
        camera_service = CameraService(performance_mode="balanced")
        
        if not camera_service.initialize_camera():
            print("Failed to initialize camera")
            return
        
        print("Camera initialized. Press 'q' to quit.")
        print("Performance modes: 'f' (fast), 'b' (balanced), 'q' (quality)")
        
        while True:
            # Capture frame
            frame = camera_service.capture_frame()
            if frame is None:
                continue
            
            # Display frame
            cv2.imshow('AI Fitness Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                ai_analyzer.set_performance_mode("fast")
                camera_service.set_performance_mode("fast")
                print("Switched to FAST mode")
            elif key == ord('b'):
                ai_analyzer.set_performance_mode("balanced")
                camera_service.set_performance_mode("balanced")
                print("Switched to BALANCED mode")
            elif key == ord('q'):
                ai_analyzer.set_performance_mode("quality")
                camera_service.set_performance_mode("quality")
                print("Switched to QUALITY mode")
            elif key == ord('a'):  # Analyze current frame
                print("Analyzing frame...")
                result = ai_analyzer.analyzeAndGenerate(frame)
                print("Analysis result:", result)
        
        # Cleanup
        camera_service.release_camera()
        cv2.destroyAllWindows()
        print("AI Analysis completed.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed.")
    except Exception as e:
        print(f"Error during AI analysis: {e}")

def run_workout_analytics():
    """Run workout analytics demonstration"""
    try:
        from ai_fitness.core.workout_analytics import WorkoutAnalytics
        
        print("Running Workout Analytics Demonstration...")
        
        # Initialize analytics
        analytics = WorkoutAnalytics()
        
        # Generate sample data if needed
        if analytics.df.empty:
            print("No workout data found. Creating sample data...")
            from ai_fitness.core.data_processing import DataProcessor
            processor = DataProcessor()
            sample_data = processor.create_sample_workout_data()
            processor.save_workout_data(sample_data)
            analytics.load_data()
        
        # Run analytics
        print("Generating workout report...")
        analytics.generate_workout_report()
        
        print("Analytics demonstration completed!")
        
    except Exception as e:
        print(f"Error running workout analytics: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are installed"""
    # Map package names to their actual import names
    package_imports = {
        'streamlit': 'streamlit',
        'opencv-contrib-python': 'cv2',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing dependencies using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed!")
    return True

def show_project_info():
    """Display project information"""
    print("=" * 60)
    print("🤖 AI FITNESS COACH")
    print("=" * 60)
    print("A comprehensive AI-powered fitness coaching application")
    print("that provides personalized workout analysis, real-time")
    print("form correction, and fitness tracking.")
    print("\nFeatures:")
    print("• AI-powered physique analysis")
    print("• Real-time pose detection and exercise tracking")
    print("• Personalized workout plan generation")
    print("• Comprehensive workout analytics and progress tracking")
    print("• Modern Streamlit web interface")
    print("\nProject Structure:")
    print("ai_fitness/")
    print("├── config/          # Configuration and settings")
    print("├── core/            # Core business logic")
    print("├── data/            # Data access and management")
    print("├── services/        # External services (camera, etc.)")
    print("├── ui/              # Streamlit user interface")
    print("└── utils/           # Utility functions and helpers")
    print("\nFor more information, see README.md")
    print("=" * 60)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="AI Fitness Coach - Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run Streamlit app
  python run_app.py --streamlit        # Run Streamlit app
  python run_app.py --tests            # Run test suite
  python run_app.py --ai-analysis      # Run AI analysis demo
  python run_app.py --analytics        # Run workout analytics demo
  python run_app.py --check-deps       # Check dependencies
  python run_app.py --info             # Show project information
        """
    )
    
    parser.add_argument(
        '--streamlit', 
        action='store_true', 
        help='Run the Streamlit web application'
    )
    parser.add_argument(
        '--tests', 
        action='store_true', 
        help='Run the test suite'
    )
    parser.add_argument(
        '--ai-analysis', 
        action='store_true', 
        help='Run AI analysis demonstration'
    )
    parser.add_argument(
        '--analytics', 
        action='store_true', 
        help='Run workout analytics demonstration'
    )
    parser.add_argument(
        '--check-deps', 
        action='store_true', 
        help='Check if all required dependencies are installed'
    )
    parser.add_argument(
        '--info', 
        action='store_true', 
        help='Show project information'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, default to running Streamlit
    if not any(vars(args).values()):
        args.streamlit = True
    
    try:
        if args.info:
            show_project_info()
        elif args.check_deps:
            check_dependencies()
        elif args.tests:
            run_tests()
        elif args.ai_analysis:
            run_ai_analysis()
        elif args.analytics:
            run_workout_analytics()
        elif args.streamlit:
            run_streamlit()
            
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 