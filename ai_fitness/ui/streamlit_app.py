"""
Streamlit UI application for AI Fitness Coach.

This module provides the main user interface for the fitness coaching
application with real-time pose detection and workout analysis.
"""

# Ensure project root is on sys.path for module imports when launched via Streamlit
import sys
from pathlib import Path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
import threading
import time
import queue
import json
from typing import Dict, Any

from ai_fitness.core.workout_analytics import WorkoutAnalytics
from ai_fitness.core.ai_analyzer import PhysiqueAnalyzer, WorkoutGenerator
from ai_fitness.services.camera_service import CameraService
from ai_fitness.config.settings import get_settings

# Page configuration
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ai-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .workout-day {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .diet-plan {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FitnessCoach:
    """Main fitness coaching class with AI analysis capabilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Performance mode selection
        self.performance_mode = st.session_state.get('performance_mode', 'balanced')
        
        # Initialize pose detection with performance mode
        self._setup_pose_detection()
        
        self.camera_service = None
        self.is_running = False
        
        # Keep only the latest frame to avoid lag
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_frame = None
        self.target_fps = 15  # Reduced from 30 for better performance
        
        # Exercise tracking
        self.exercise_stage = "up"
        self.rep_count = 0
        self.last_rep_time = 0
        self.rep_cooldown = 1.0  # 1 second cooldown between reps
        
        # Performance monitoring
        self.frame_times = []
        self.processing_times = []
        
        # AI components
        self.physique_analyzer = PhysiqueAnalyzer()
        self.workout_generator = WorkoutGenerator()
        self.workout_analytics = WorkoutAnalytics()
    
    def _setup_pose_detection(self):
        """Setup pose detection based on performance mode"""
        if self.performance_mode == "fast":
            min_detection_confidence = 0.5
            min_tracking_confidence = 0.5
            model_complexity = 0
        elif self.performance_mode == "balanced":
            min_detection_confidence = 0.6
            min_tracking_confidence = 0.6
            model_complexity = 1
        else:  # quality mode
            min_detection_confidence = 0.7
            min_tracking_confidence = 0.7
            model_complexity = 2
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False
        )
        
        print(f"Pose detection setup with {self.performance_mode} mode")
        print(f"Model complexity: {model_complexity}, Detection confidence: {min_detection_confidence}")
    
    def set_performance_mode(self, mode: str):
        """Change performance mode and reinitialize components"""
        if mode in ["fast", "balanced", "quality"]:
            self.performance_mode = mode
            st.session_state.performance_mode = mode
            
            # Reinitialize pose detection
            self._setup_pose_detection()
            
            # Update camera service if running
            if self.camera_service:
                self.camera_service.set_performance_mode(mode)
            
            st.success(f"Performance mode changed to: {mode}")
        else:
            st.error(f"Invalid performance mode: {mode}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            'performance_mode': self.performance_mode,
            'target_fps': self.target_fps,
            'is_running': self.is_running,
            'frame_queue_size': self.frame_queue.qsize()
        }
        
        if self.camera_service:
            stats.update(self.camera_service.get_performance_stats())
        
        if self.frame_times:
            stats['avg_frame_time'] = sum(self.frame_times) / len(self.frame_times)
            stats['total_frames_processed'] = len(self.frame_times)
        
        return stats
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def start_camera(self):
        """Start optimized camera service for pose detection"""
        try:
            if self.camera_service is None:
                # Initialize camera service with current performance mode
                self.camera_service = CameraService(performance_mode=self.performance_mode)
                if not self.camera_service.initialize_camera():
                    st.error("Failed to initialize camera")
                    return False
            
            self.is_running = True
            
            # Start streaming with frame callback
            self.camera_service.start_streaming(callback=self._process_frame)
            st.success(f"Camera started successfully with {self.performance_mode} mode!")
            return True
            
        except Exception as e:
            st.error(f"Error starting camera: {str(e)}")
            return False
    
    def _process_frame(self, frame):
        """Process incoming camera frame"""
        try:
            # Update latest frame
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # Process pose detection
            self._detect_pose(frame)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
    
    def _detect_pose(self, frame):
        """Detect pose in the frame and track exercises"""
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Track push-ups
                self._track_pushups(results.pose_landmarks, frame)
                
                # Track squats
                self._track_squats(results.pose_landmarks, frame)
                
        except Exception as e:
            print(f"Error in pose detection: {str(e)}")
    
    def _track_pushups(self, landmarks, frame):
        """Track push-up repetitions"""
        try:
            # Get relevant landmarks
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate angles
            left_angle = self.calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            )
            
            right_angle = self.calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            # Track push-up state
            current_time = time.time()
            if left_angle < 90 and right_angle < 90:
                if self.exercise_stage == "up":
                    self.exercise_stage = "down"
            elif left_angle > 160 and right_angle > 160:
                if self.exercise_stage == "down" and (current_time - self.last_rep_time) > self.rep_cooldown:
                    self.exercise_stage = "up"
                    self.rep_count += 1
                    self.last_rep_time = current_time
            
            # Display rep count
            cv2.putText(frame, f'Push-ups: {self.rep_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error tracking push-ups: {str(e)}")
    
    def _track_squats(self, landmarks, frame):
        """Track squat repetitions"""
        try:
            # Get relevant landmarks
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Calculate angles
            left_angle = self.calculate_angle(
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y]
            )
            
            right_angle = self.calculate_angle(
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y]
            )
            
            # Display squat angle
            cv2.putText(frame, f'Squat Angle: {int(left_angle)}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        except Exception as e:
            print(f"Error tracking squats: {str(e)}")
    
    def test_camera_access(self):
        """Test camera access without starting full stream"""
        try:
            if self.camera_service is None:
                self.camera_service = CameraService()
            
            if self.camera_service.initialize_camera():
                # Take a test photo
                test_photo = self.camera_service.take_photo("test_camera_access.jpg")
                if test_photo:
                    st.success("Camera access test successful!")
                    return True
                else:
                    st.error("Camera access test failed - could not capture image")
                    return False
            else:
                st.error("Camera access test failed - could not initialize camera")
                return False
                
        except Exception as e:
            st.error(f"Camera access test error: {str(e)}")
            return False
    
    def stop_camera(self):
        """Stop camera service"""
        try:
            if self.camera_service:
                self.camera_service.stop_streaming()
                self.camera_service.release_camera()
                self.camera_service = None
            
            self.is_running = False
            st.success("Camera stopped successfully!")
            
        except Exception as e:
            st.error(f"Error stopping camera: {str(e)}")
    
    def capture_frames(self):
        """Capture frames for pose analysis"""
        try:
            if not self.camera_service or not self.is_running:
                return None
            
            # Get latest frame
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                return frame
            
            return None
            
        except Exception as e:
            print(f"Error capturing frames: {str(e)}")
            return None
    
    def process_frame(self, frame):
        """Process frame for display and analysis"""
        try:
            if frame is None:
                return None
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add exercise tracking overlay
            if self.rep_count > 0:
                cv2.putText(frame_rgb, f'Total Reps: {self.rep_count}', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return frame_rgb
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

def main():
    """Main Streamlit application"""
    # Initialize session state variables
    if 'fitness_coach' not in st.session_state:
        st.session_state.fitness_coach = FitnessCoach()
    
    if 'camera_started' not in st.session_state:
        st.session_state.camera_started = False
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Main header
    st.markdown('<h1 class="main-header">💪 AI Fitness Coach</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📷 AI Analysis", "📊 Workout Analytics", "🎯 Workout Plans", "⚙️ Settings"]
    )
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📷 AI Analysis":
        show_ai_analysis_page()
    elif page == "📊 Workout Analytics":
        show_workout_analytics_page()
    elif page == "🎯 Workout Plans":
        show_workout_plans_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page with overview and quick actions"""
    st.header("Welcome to AI Fitness Coach! 🚀")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AI Analysis", "Ready", delta="Active")
    
    with col2:
        st.metric("Workout Tracking", "Active", delta="+5 reps")
    
    with col3:
        st.metric("Progress", "On Track", delta="+2.5%")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Start AI Analysis", use_container_width=True):
            st.session_state.page = "📷 AI Analysis"
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Workout Analytics"
            st.rerun()
    
    # Recent activity
    st.subheader("Recent Activity")
    st.info("No recent workouts. Start your fitness journey today!")

def show_ai_analysis_page():
    """Display the AI analysis page with camera and pose detection"""
    st.header("🤖 AI-Powered Fitness Analysis")
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📷 Start Camera", use_container_width=True):
            if st.session_state.fitness_coach.start_camera():
                st.session_state.camera_started = True
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.fitness_coach.stop_camera()
            st.session_state.camera_started = False
            st.rerun()
    
    with col3:
        if st.button("🧪 Test Camera", use_container_width=True):
            st.session_state.fitness_coach.test_camera_access()
    
    # Performance mode controls
    st.subheader("Performance Mode")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        performance_mode = st.radio(
            "Select Performance Mode:",
            ["fast", "balanced", "quality"],
            index=["fast", "balanced", "quality"].index(st.session_state.fitness_coach.performance_mode),
            horizontal=True,
            help="Fast: Lower resolution, higher speed. Balanced: Medium resolution and speed. Quality: Higher resolution, lower speed."
        )
        
        if performance_mode != st.session_state.fitness_coach.performance_mode:
            st.session_state.fitness_coach.set_performance_mode(performance_mode)
            st.rerun()
    
    with col2:
        if st.button("📊 Performance Stats", use_container_width=True):
            stats = st.session_state.fitness_coach.get_performance_stats()
            st.json(stats)
    
    # Performance information
    with st.expander("ℹ️ Performance Mode Information"):
        st.markdown("""
        **Fast Mode**: 
        - Resolution: 480x360
        - Target FPS: 15
        - Model Complexity: Lightweight
        - Best for: Real-time tracking, lower-end devices
        
        **Balanced Mode**: 
        - Resolution: 640x480
        - Target FPS: 20
        - Model Complexity: Medium
        - Best for: General use, good balance of speed and quality
        
        **Quality Mode**: 
        - Resolution: 1280x720
        - Target FPS: 30
        - Model Complexity: Full
        - Best for: Detailed analysis, high-end devices
        """)
    
    # Camera feed
    if st.session_state.camera_started:
        st.subheader("Live Camera Feed")
        
        # Camera placeholder
        camera_placeholder = st.empty()
        
        # Exercise tracking display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Push-ups", f"{st.session_state.fitness_coach.rep_count}")
        
        with col2:
            st.metric("Exercise Stage", st.session_state.fitness_coach.exercise_stage.title())
        
        # Real-time frame processing
        if st.session_state.fitness_coach.is_running:
            frame = st.session_state.fitness_coach.capture_frames()
            if frame is not None:
                processed_frame = st.session_state.fitness_coach.process_frame(frame)
                if processed_frame is not None:
                    camera_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
    
    else:
        st.info("Click 'Start Camera' to begin AI analysis")
    
    # AI Analysis Results
    if st.session_state.analysis_results:
        st.subheader("Analysis Results")
        st.json(st.session_state.analysis_results)

def show_workout_analytics_page():
    """Display workout analytics and progress tracking"""
    st.header("📊 Workout Analytics & Progress")
    
    # Generate sample data if needed
    if st.button("📈 Generate Sample Data", use_container_width=True):
        st.session_state.fitness_coach.workout_analytics.create_sample_workout_data()
        st.success("Sample data generated!")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Daily Activity", "📊 Exercise Comparison", "📅 Weekly Trends", "🔥 Activity Heatmap"])
    
    with tab1:
        st.subheader("Daily Exercise Activity")
        if st.button("Generate Daily Activity Chart"):
            st.session_state.fitness_coach.workout_analytics.plot_daily_activity()
    
    with tab2:
        st.subheader("Exercise Performance Comparison")
        if st.button("Generate Exercise Comparison"):
            st.session_state.fitness_coach.workout_analytics.plot_exercise_comparison()
    
    with tab3:
        st.subheader("Weekly Exercise Trends")
        if st.button("Generate Weekly Trends"):
            st.session_state.fitness_coach.workout_analytics.plot_weekly_trends()
    
    with tab4:
        st.subheader("Weekly Activity Heatmap")
        if st.button("Generate Activity Heatmap"):
            st.session_state.fitness_coach.workout_analytics.create_heatmap()
    
    # Workout Report
    if st.button("📋 Generate Workout Report", use_container_width=True):
        st.session_state.fitness_coach.workout_analytics.generate_workout_report()

def show_workout_plans_page():
    """Display workout plans and recommendations"""
    st.header("🎯 Personalized Workout Plans")
    
    st.info("AI-generated workout plans will appear here based on your analysis results.")
    
    # Placeholder for workout plans
    st.subheader("Your Workout Plan")
    st.write("Complete an AI analysis to get your personalized workout plan!")

def show_settings_page():
    """Display application settings and configuration"""
    st.header("⚙️ Application Settings")
    
    st.subheader("Camera Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
        camera_width = st.number_input("Camera Width", min_value=320, max_value=1920, value=640, step=160)
    
    with col2:
        camera_height = st.number_input("Camera Height", min_value=240, max_value=1080, value=480, step=120)
        camera_fps = st.number_input("Camera FPS", min_value=1, max_value=60, value=30)
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")
    
    st.subheader("AI Model Settings")
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.success("Settings reset to defaults!")

if __name__ == "__main__":
    main()
