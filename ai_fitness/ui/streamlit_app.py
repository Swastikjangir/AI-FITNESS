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
# Try to import OpenCV; gracefully degrade on platforms without cv2 support (e.g., Streamlit Cloud)
try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover
    OPENCV_AVAILABLE = False
    CV2_IMPORT_ERROR = str(_e)
# Import MediaPipe only if OpenCV is available; mediapipe imports cv2 internally
if OPENCV_AVAILABLE:
    try:
        import mediapipe as mp  # type: ignore
        MEDIAPIPE_AVAILABLE = True
        MP_IMPORT_ERROR = ""
    except Exception as _e:  # pragma: no cover
        MEDIAPIPE_AVAILABLE = False
        MP_IMPORT_ERROR = str(_e)
        mp = None  # type: ignore
else:
    MEDIAPIPE_AVAILABLE = False
    MP_IMPORT_ERROR = "OpenCV not available; skipping MediaPipe import"
    mp = None  # type: ignore
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
        self.mp_drawing = mp.solutions.drawing_utils if MEDIAPIPE_AVAILABLE else None
        self.mp_pose = mp.solutions.pose if MEDIAPIPE_AVAILABLE else None
        
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
        self.squat_count = 0
        self.last_rep_time = 0
        self.rep_cooldown = 1.0  # 1 second cooldown between reps
        
        # Workout session tracking
        self.workout_start_time = None
        self.workout_duration = 0
        
        # Performance monitoring
        self.frame_times = []
        self.processing_times = []
        
        # AI components
        self.physique_analyzer = PhysiqueAnalyzer()
        self.workout_generator = WorkoutGenerator()
        self.workout_analytics = WorkoutAnalytics()
    
    def _setup_pose_detection(self):
        """Setup pose detection based on performance mode"""
        if not MEDIAPIPE_AVAILABLE or self.mp_pose is None:
            self.pose = None
            return
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
            if not OPENCV_AVAILABLE:
                st.error("OpenCV (cv2) is not available in this environment. Live camera is disabled.\n" \
                         f"Import error: {CV2_IMPORT_ERROR}")
                return False
            # Import lazily here to avoid import errors during app startup on cloud
            from ai_fitness.services.camera_service import CameraService
            if self.camera_service is None:
                # Initialize camera service with current performance mode
                self.camera_service = CameraService(performance_mode=self.performance_mode)
                if not self.camera_service.initialize_camera():
                    st.error("Failed to initialize camera")
                    return False
            
            self.is_running = True
            
            # Start streaming with frame callback
            self.camera_service.start_streaming(callback=self._process_frame)
            
            # Start workout session
            self.start_workout_session()
            
            # Clear previous analysis results
            if hasattr(st, 'session_state') and 'analysis_results' in st.session_state:
                st.session_state.analysis_results = None
            
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
            if not OPENCV_AVAILABLE or not MEDIAPIPE_AVAILABLE or self.pose is None:
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                if self.mp_drawing and self.mp_pose:
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
            if not MEDIAPIPE_AVAILABLE or self.mp_pose is None:
                return
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
            if not OPENCV_AVAILABLE:
                return
            cv2.putText(frame, f'Push-ups: {self.rep_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error tracking push-ups: {str(e)}")
    
    def _track_squats(self, landmarks, frame):
        """Track squat repetitions"""
        try:
            # Get relevant landmarks
            if not MEDIAPIPE_AVAILABLE or self.mp_pose is None:
                return
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
            
            # Track squat state and count reps
            current_time = time.time()
            avg_angle = (left_angle + right_angle) / 2
            
            if avg_angle < 110:  # Squat down position
                if self.exercise_stage == "up":
                    self.exercise_stage = "down"
            elif avg_angle > 160:  # Standing position
                if self.exercise_stage == "down" and (current_time - self.last_rep_time) > self.rep_cooldown:
                    self.exercise_stage = "up"
                    self.squat_count += 1
                    self.last_rep_time = current_time
            
            # Display squat count
            if not OPENCV_AVAILABLE:
                return
            cv2.putText(frame, f'Squats: {self.squat_count}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        except Exception as e:
            print(f"Error tracking squats: {str(e)}")
    
    def test_camera_access(self):
        """Test camera access without starting full stream"""
        try:
            if self.camera_service is None:
                from ai_fitness.services.camera_service import CameraService
                self.camera_service = CameraService()
            
            if self.camera_service.initialize_camera():
                # Test frame capture instead of photo
                test_frame = self.camera_service.capture_frame()
                if test_frame is not None:
                    st.success("Camera access test successful! Camera is working properly.")
                    return True
                else:
                    st.error("Camera access test failed - could not capture frame")
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
            if not OPENCV_AVAILABLE or (not self.camera_service or not self.is_running):
                return None
            
            # Get latest frame from camera service
            if self.camera_service:
                return self.camera_service.get_latest_frame()
            
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
            if not OPENCV_AVAILABLE:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add exercise tracking overlay
            if self.rep_count > 0 or self.squat_count > 0:
                cv2.putText(frame_rgb, f'Push-ups: {self.rep_count}', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_rgb, f'Squats: {self.squat_count}', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            return frame_rgb
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None
    
    def reset_workout(self):
        """Reset workout counters and state"""
        self.rep_count = 0
        self.squat_count = 0
        self.exercise_stage = "up"
        self.last_rep_time = 0
        self.workout_start_time = None
        self.workout_duration = 0
        print("Workout counters reset")
    
    def start_workout_session(self):
        """Start a new workout session"""
        self.workout_start_time = time.time()
        print("Workout session started")
    
    def end_workout_session(self):
        """End the current workout session and calculate duration"""
        if self.workout_start_time:
            self.workout_duration = time.time() - self.workout_start_time
            print(f"Workout session ended. Duration: {self.workout_duration:.1f} seconds")
            return self.workout_duration
        return 0
    
    def get_current_workout_duration(self):
        """Get the current workout duration in seconds"""
        if self.workout_start_time:
            return time.time() - self.workout_start_time
        return 0
    
    def get_workout_status(self):
        """Get current workout status summary"""
        return {
            'is_running': self.is_running,
            'pushup_count': self.rep_count,
            'squat_count': self.squat_count,
            'total_reps': self.rep_count + self.squat_count,
            'duration': self.get_current_workout_duration(),
            'exercise_stage': self.exercise_stage,
            'performance_mode': self.performance_mode
        }

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
    
    # Real-time workout status
    if st.session_state.fitness_coach.is_running:
        st.success("🔥 **LIVE WORKOUT IN PROGRESS**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Push-ups", f"{st.session_state.fitness_coach.rep_count}", delta="Live")
        
        with col2:
            st.metric("Squats", f"{st.session_state.fitness_coach.squat_count}", delta="Live")
        
        with col3:
            current_duration = st.session_state.fitness_coach.get_current_workout_duration()
            st.metric("Duration", f"{int(current_duration)}s", delta="Live")
        
        with col4:
            total_reps = st.session_state.fitness_coach.rep_count + st.session_state.fitness_coach.squat_count
            st.metric("Total Reps", total_reps, delta="Live")
        
        # Quick workout actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Analyze Current Workout", use_container_width=True, type="primary"):
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Workout", use_container_width=True):
                st.session_state.fitness_coach.stop_camera()
                st.session_state.camera_started = False
                st.rerun()
    
    # General metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
            total_sessions = len(st.session_state.fitness_coach.workout_analytics.df['date'].unique())
            st.metric("Total Sessions", total_sessions, delta="+1" if st.session_state.fitness_coach.is_running else None)
        else:
            st.metric("Total Sessions", "0", delta="Start Today!")
    
    with col2:
        if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
            total_reps = st.session_state.fitness_coach.workout_analytics.df['count'].sum()
            st.metric("Total Reps", total_reps, delta="+1" if st.session_state.fitness_coach.is_running else None)
        else:
            st.metric("Total Reps", "0", delta="Start Today!")
    
    with col3:
        if st.session_state.fitness_coach.is_running:
            st.metric("Status", "Active", delta="Live")
        else:
            st.metric("Status", "Ready", delta="Start Workout")
    
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
    if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
        recent_data = st.session_state.fitness_coach.workout_analytics.df.tail(5)
        for _, row in recent_data.iterrows():
            st.info(f"✅ {row['exercise'].title()}: {row['count']} reps on {row['timestamp']}")
    else:
        st.info("No recent workouts. Start your fitness journey today!")

def show_ai_analysis_page():
    """Display the AI analysis page with camera and pose detection"""
    st.header("🤖 AI-Powered Fitness Analysis")
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📷 Start Camera", use_container_width=True):
            if not OPENCV_AVAILABLE:
                st.warning("Live camera is not available on this platform (likely Streamlit Cloud). Try uploading a photo below.")
            else:
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
            if not OPENCV_AVAILABLE:
                st.warning("Camera test is not available on this platform.")
            else:
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
    if OPENCV_AVAILABLE and st.session_state.camera_started:
        st.subheader("Live Camera Feed")
        
        # Camera placeholder
        camera_placeholder = st.empty()
        
        # Exercise tracking display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Push-ups", f"{st.session_state.fitness_coach.rep_count}")
        
        with col2:
            st.metric("Squats", f"{st.session_state.fitness_coach.squat_count}")
        
        with col3:
            st.metric("Exercise Stage", st.session_state.fitness_coach.exercise_stage.title())
        
        with col4:
            current_duration = st.session_state.fitness_coach.get_current_workout_duration()
            st.metric("Workout Time", f"{int(current_duration)}s")
        
        # Analysis button
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔍 Analyze Workout", use_container_width=True, type="primary"):
                # Generate workout report based on current session
                total_reps = st.session_state.fitness_coach.rep_count + st.session_state.fitness_coach.squat_count
                if total_reps > 0:
                    # End workout session and get duration
                    duration = st.session_state.fitness_coach.end_workout_session()
                    
                    # Save workout data for both exercises
                    workout_data_list = []
                    
                    if st.session_state.fitness_coach.rep_count > 0:
                        pushup_data = st.session_state.fitness_coach.workout_analytics.save_workout_data(
                            exercise='push_ups',
                            count=st.session_state.fitness_coach.rep_count,
                            duration=int(duration),
                            calories=st.session_state.fitness_coach.rep_count * 0.5
                        )
                        workout_data_list.append(pushup_data)
                    
                    if st.session_state.fitness_coach.squat_count > 0:
                        squat_data = st.session_state.fitness_coach.workout_analytics.save_workout_data(
                            exercise='squats',
                            count=st.session_state.fitness_coach.squat_count,
                            duration=int(duration),
                            calories=st.session_state.fitness_coach.squat_count * 0.3
                        )
                        workout_data_list.append(squat_data)
                    
                    # Get workout summary
                    workout_summary = st.session_state.fitness_coach.workout_analytics.get_workout_summary(
                        st.session_state.fitness_coach.rep_count,
                        st.session_state.fitness_coach.squat_count,
                        int(duration)
                    )
                    
                    st.success(f"Workout analyzed! {st.session_state.fitness_coach.rep_count} push-ups and {st.session_state.fitness_coach.squat_count} squats in {workout_summary['duration_minutes']} minutes recorded.")
                    st.session_state.analysis_results = workout_summary
                else:
                    st.warning("No workout data to analyze. Please perform some exercises first.")
        
        # Reset button
        with col3:
            if st.button("🔄 Reset Counter", use_container_width=True):
                st.session_state.fitness_coach.reset_workout()
                # Clear analysis results
                st.session_state.analysis_results = None
                st.success("Workout counter reset!")
                st.rerun()
        
        # Real-time video streaming using Streamlit's rerun mechanism
        if st.session_state.fitness_coach.is_running:
            # Use the new streaming method for better performance
            if not hasattr(st.session_state, 'streaming_thread') or not st.session_state.streaming_thread.is_alive():
                # Create a processing callback for pose detection
                def process_frame_with_pose(frame):
                    """Process frame with pose detection and exercise tracking"""
                    try:
                        # Process pose detection
                        st.session_state.fitness_coach._detect_pose(frame)
                        
                        # Add exercise tracking overlay
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if st.session_state.fitness_coach.rep_count > 0 or st.session_state.fitness_coach.squat_count > 0:
                            cv2.putText(frame_rgb, f'Push-ups: {st.session_state.fitness_coach.rep_count}', (10, 110), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame_rgb, f'Squats: {st.session_state.fitness_coach.squat_count}', (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
                        return frame_rgb
                    except Exception as e:
                        print(f"Error in pose processing: {str(e)}")
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Start streaming with pose processing
                st.session_state.streaming_thread = threading.Thread(
                    target=st.session_state.fitness_coach.camera_service.stream_frames_streamlit_with_processing,
                    args=(camera_placeholder, process_frame_with_pose, None, 20)  # 20 FPS for smooth tracking
                )
                st.session_state.streaming_thread.daemon = True
                st.session_state.streaming_thread.start()
            
            # Show streaming status
            st.success("🎥 Video streaming with pose detection is active!")
            
            # Performance stats
            if st.session_state.fitness_coach.camera_service:
                stats = st.session_state.fitness_coach.camera_service.get_performance_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mode", stats.get('performance_mode', 'N/A').title())
                with col2:
                    st.metric("Resolution", stats.get('resolution', 'N/A'))
                with col3:
                    st.metric("Target FPS", stats.get('target_fps', 'N/A'))
    
    else:
        # Cloud-safe input
        st.info("Live camera not available here. Upload a snapshot to analyze.")
        uploaded = st.file_uploader("Upload an image (jpg/png)")
        if uploaded is not None and OPENCV_AVAILABLE:
            file_bytes = uploaded.read()
            import numpy as _np
            img_arr = _np.frombuffer(file_bytes, dtype=_np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is not None:
                processed = st.session_state.fitness_coach.process_frame(img)
                if processed is not None:
                    st.image(processed, channels="RGB", use_container_width=True)
        elif uploaded is not None and not OPENCV_AVAILABLE:
            st.warning("OpenCV not available; image processing is disabled in this environment.")
    
    # AI Analysis Results
    if st.session_state.analysis_results:
        st.subheader("📊 Workout Analysis Results")
        
        # Display results in a nice format
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reps", st.session_state.analysis_results['total_reps'])
        
        with col2:
            st.metric("Duration", f"{st.session_state.analysis_results['duration_minutes']} min")
        
        with col3:
            st.metric("Calories Burned", f"{st.session_state.analysis_results['total_calories']}")
        
        with col4:
            st.metric("Timestamp", st.session_state.analysis_results['timestamp'])
        
        # Detailed breakdown
        st.subheader("Exercise Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Push-ups", f"{st.session_state.analysis_results['push_ups']} reps")
            st.metric("Push-up Calories", f"{st.session_state.analysis_results['pushup_calories']}")
        
        with col2:
            st.metric("Squats", f"{st.session_state.analysis_results['squats']} reps")
            st.metric("Squat Calories", f"{st.session_state.analysis_results['squat_calories']}")
        
        # Raw data (collapsible)
        with st.expander("📋 Raw Analysis Data"):
            st.json(st.session_state.analysis_results)

def show_workout_analytics_page():
    """Display workout analytics and progress tracking"""
    st.header("📊 Workout Analytics & Progress")
    
    # Current workout session summary
    if st.session_state.fitness_coach.is_running:
        st.subheader("🔥 Current Workout Session")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Push-ups", f"{st.session_state.fitness_coach.rep_count}")
        
        with col2:
            st.metric("Squats", f"{st.session_state.fitness_coach.squat_count}")
        
        with col3:
            current_duration = st.session_state.fitness_coach.get_current_workout_duration()
            st.metric("Duration", f"{int(current_duration)}s")
        
        with col4:
            total_reps = st.session_state.fitness_coach.rep_count + st.session_state.fitness_coach.squat_count
            st.metric("Total Reps", total_reps)
        
        # Quick actions for current session
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Analyze Current Workout", use_container_width=True, type="primary"):
                if total_reps > 0:
                    # End workout session and get duration
                    duration = st.session_state.fitness_coach.end_workout_session()
                    
                    # Save workout data for both exercises
                    if st.session_state.fitness_coach.rep_count > 0:
                        st.session_state.fitness_coach.workout_analytics.save_workout_data(
                            exercise='push_ups',
                            count=st.session_state.fitness_coach.rep_count,
                            duration=int(duration),
                            calories=st.session_state.fitness_coach.rep_count * 0.5
                        )
                    
                    if st.session_state.fitness_coach.squat_count > 0:
                        st.session_state.fitness_coach.workout_analytics.save_workout_data(
                            exercise='squats',
                            count=st.session_state.fitness_coach.squat_count,
                            duration=int(duration),
                            calories=st.session_state.fitness_coach.squat_count * 0.3
                        )
                    
                    st.success(f"Workout saved! {st.session_state.fitness_coach.rep_count} push-ups and {st.session_state.fitness_coach.squat_count} squats recorded.")
                    st.rerun()
                else:
                    st.warning("No exercises performed yet!")
        
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.fitness_coach.reset_workout()
                st.success("Session reset!")
                st.rerun()
    
    # Generate sample data if needed
    if st.button("📈 Generate Sample Data", use_container_width=True):
        st.session_state.fitness_coach.workout_analytics.create_sample_workout_data()
        st.success("Sample data generated!")
    
    # Refresh data button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.fitness_coach.workout_analytics.load_data()
        st.success("Data refreshed!")
        st.rerun()
    
    # Analytics tabs
    tab1, tab2 = st.tabs(["📊 Exercise Summary", "📈 Progress Data"])
    
    with tab1:
        st.subheader("Exercise Summary")
        if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
            # Show exercise totals
            exercise_totals = st.session_state.fitness_coach.workout_analytics.df.groupby('exercise')['count'].sum().reset_index()
            st.dataframe(exercise_totals, use_container_width=True)
        else:
            st.info("No workout data available. Complete some workouts to see your summary!")
    
    with tab2:
        st.subheader("Progress Data")
        if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
            # Show recent daily data
            daily_summary = st.session_state.fitness_coach.workout_analytics.get_daily_summary()
            if not daily_summary.empty:
                st.dataframe(daily_summary.tail(7), use_container_width=True)
        else:
            st.info("No progress data available. Complete some workouts to see your progress!")
    
    # Workout Report
    if st.button("📋 Generate Workout Report", use_container_width=True):
        st.session_state.fitness_coach.workout_analytics.generate_workout_report()

def show_workout_plans_page():
    """Display workout plans and recommendations"""
    st.header("🎯 Personalized Workout Plans")
    
    # Check if user has workout data
    if st.session_state.fitness_coach.workout_analytics.df is not None and not st.session_state.fitness_coach.workout_analytics.df.empty:
        st.success("Great! We have workout data to create personalized plans.")
        
        # Get recent workout summary
        recent_data = st.session_state.fitness_coach.workout_analytics.df.tail(10)
        total_recent_reps = recent_data['count'].sum()
        avg_reps_per_session = recent_data.groupby('exercise')['count'].mean()
        
        # Display current fitness level
        st.subheader("📊 Your Current Fitness Level")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recent Sessions", len(recent_data))
        
        with col2:
            st.metric("Total Recent Reps", total_recent_reps)
        
        with col3:
            st.metric("Exercises Tracked", len(avg_reps_per_session))
        
        # Personalized recommendations
        st.subheader("🎯 Personalized Recommendations")
        
        if total_recent_reps < 50:
            st.info("**Beginner Level** - Focus on building consistency and proper form")
            st.markdown("""
            **Recommended Workout Plan:**
            - **Day 1**: 3 sets of 5-10 push-ups, 3 sets of 5-10 squats
            - **Day 2**: Rest or light stretching
            - **Day 3**: 3 sets of 5-10 lunges, 3 sets of 30-second planks
            - **Day 4**: Rest
            - **Day 5**: Repeat Day 1
            - **Goal**: Build up to 20 reps per exercise
            """)
        
        elif total_recent_reps < 150:
            st.info("**Intermediate Level** - Time to increase intensity and variety")
            st.markdown("""
            **Recommended Workout Plan:**
            - **Day 1**: 4 sets of 15-20 push-ups, 4 sets of 15-20 squats
            - **Day 2**: 4 sets of 15-20 lunges, 4 sets of 60-second planks
            - **Day 3**: Rest or active recovery (walking, stretching)
            - **Day 4**: 4 sets of 15-20 burpees, 4 sets of 15-20 mountain climbers
            - **Day 5**: Repeat Day 1
            - **Goal**: Build up to 30 reps per exercise
            """)
        
        else:
            st.success("**Advanced Level** - Excellent progress! Time for challenging variations")
            st.markdown("""
            **Recommended Workout Plan:**
            - **Day 1**: 5 sets of 25-30 push-ups, 5 sets of 25-30 squats
            - **Day 2**: 5 sets of 25-30 lunges, 5 sets of 90-second planks
            - **Day 3**: 5 sets of 25-30 burpees, 5 sets of 25-30 mountain climbers
            - **Day 4**: Active recovery with yoga or swimming
            - **Day 5**: High-intensity interval training (HIIT)
            - **Goal**: Maintain high reps and add weight/resistance
            """)
        
        # Progress tracking
        st.subheader("📈 Progress Tracking")
        if st.button("🔄 Update Progress Analysis"):
            st.session_state.fitness_coach.workout_analytics.generate_workout_report()
            st.success("Progress analysis updated!")
        
        # Exercise variety suggestions
        st.subheader("💡 Try New Exercises")
        st.markdown("""
        **Based on your current routine, consider adding:**
        - **Cardio**: Jumping jacks, high knees, mountain climbers
        - **Strength**: Burpees, plank variations, wall sits
        - **Flexibility**: Yoga poses, dynamic stretching
        - **Balance**: Single-leg exercises, stability work
        """)
        
    else:
        st.info("Complete some AI analysis workouts to get your personalized workout plan!")
        
        # Placeholder for workout plans
        st.subheader("Sample Workout Plan")
        st.markdown("""
        **Basic Fitness Routine (Complete this to unlock personalized plans):**
        
        **Week 1-2: Foundation Building**
        - **Monday**: 3 sets of 5 push-ups, 3 sets of 5 squats
        - **Wednesday**: 3 sets of 5 lunges, 3 sets of 30-second planks
        - **Friday**: Repeat Monday's workout
        - **Weekend**: Rest and recovery
        
        **Goal**: Complete at least 3 workouts this week to unlock personalized recommendations!
        """)
        
        if st.button("🎯 Start Your Fitness Journey", use_container_width=True):
            st.info("Navigate to '📷 AI Analysis' to begin your first workout session!")

def show_settings_page():
    """Display application settings and configuration"""
    st.header("⚙️ Application Settings")
    
    # Current system status
    st.subheader("🔍 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("OpenCV Available", "✅ Yes" if OPENCV_AVAILABLE else "❌ No")
    
    with col2:
        st.metric("MediaPipe Available", "✅ Yes" if MEDIAPIPE_AVAILABLE else "❌ No")
    
    with col3:
        st.metric("Camera Status", "🟢 Active" if st.session_state.fitness_coach.is_running else "🔴 Inactive")
    
    # Current workout status
    if st.session_state.fitness_coach.is_running:
        st.subheader("🔥 Current Workout Status")
        workout_status = st.session_state.fitness_coach.get_workout_status()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Push-ups", workout_status['pushup_count'])
        with col2:
            st.metric("Squats", workout_status['squat_count'])
        with col3:
            st.metric("Duration", f"{int(workout_status['duration'])}s")
        with col4:
            st.metric("Stage", workout_status['exercise_stage'].title())
    
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
    
    # Performance information
    st.subheader("📊 Performance Information")
    if st.button("📈 Show Performance Stats"):
        stats = st.session_state.fitness_coach.get_performance_stats()
        st.json(stats)

if __name__ == "__main__":
    main()
