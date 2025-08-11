"""
Camera service module for AI Fitness Coach.

This module provides camera functionality for real-time pose detection,
workout form analysis, and video capture capabilities with performance optimizations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
import threading
import time
from pathlib import Path
import os
import queue
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp

from ai_fitness.config.settings import get_settings

class OptimizedCameraService:
    """Optimized camera service for smooth real-time video processing"""
    
    def __init__(self, camera_index: int = None, performance_mode: str = "balanced"):
        self.settings = get_settings()
        self.camera_index = camera_index or self.settings.camera_index
        self.camera = None
        self.is_recording = False
        self.is_streaming = False
        self.frame_callback = None
        self.recording_thread = None
        self.streaming_thread = None
        
        # Performance optimization settings
        self.performance_mode = performance_mode
        self._setup_performance_mode()
        
        # Frame processing optimization
        self.frame_skip = 2  # Process every Nth frame
        self.frame_count = 0
        self.last_processed_time = 0
        self.min_processing_interval = 1.0 / 15  # Max 15 FPS processing
        
        # Threading and queuing
        self.frame_queue = queue.Queue(maxsize=3)  # Small buffer to prevent lag
        self.processed_frame_queue = queue.Queue(maxsize=2)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Camera properties
        self.width = self.settings.camera_width
        self.height = self.settings.camera_height
        self.fps = self.settings.camera_fps
        
        # Recording settings
        self.output_dir = self.settings.data_dir / "recordings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_performance_mode(self):
        """Configure performance settings based on mode"""
        if self.performance_mode == "fast":
            # Prioritize speed over quality
            self.width = 480
            self.height = 360
            self.fps = 15
            self.frame_skip = 3
            self.min_processing_interval = 1.0 / 10
        elif self.performance_mode == "balanced":
            # Balance between speed and quality
            self.width = 640
            self.height = 480
            self.fps = 20
            self.frame_skip = 2
            self.min_processing_interval = 1.0 / 15
        elif self.performance_mode == "quality":
            # Prioritize quality over speed
            self.width = 1280
            self.height = 720
            self.fps = 30
            self.frame_skip = 1
            self.min_processing_interval = 1.0 / 20
        
        print(f"Performance mode: {self.performance_mode}")
        print(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}")
        
    def initialize_camera(self) -> bool:
        """Initialize camera connection with performance optimizations"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                print(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set camera properties for performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Performance optimizations
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for better performance
            
            # Verify properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
            print(f"Performance mode: {self.performance_mode}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            return False
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        print("Camera released")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera with error handling"""
        if not self.camera or not self.camera.isOpened():
            print("Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Resize frame if needed for performance
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                return frame
            else:
                return None
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
    
    def start_streaming(self, callback: Callable[[np.ndarray], None] = None):
        """Start optimized video streaming"""
        if self.is_streaming:
            print("Streaming already active")
            return
        
        self.frame_callback = callback
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._optimized_stream_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        print("Optimized streaming started")
    
    def stop_streaming(self):
        """Stop video streaming"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
        print("Streaming stopped")
    
    def _optimized_stream_loop(self):
        """Optimized streaming loop with frame skipping and async processing"""
        while self.is_streaming:
            try:
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.01)  # Short sleep if no frame
                    continue
                
                current_time = time.time()
                
                # Frame skipping for performance
                if self.frame_count % self.frame_skip != 0:
                    self.frame_count += 1
                    continue
                
                # Rate limiting for processing
                if current_time - self.last_processed_time < self.min_processing_interval:
                    time.sleep(0.001)  # Micro-sleep
                    continue
                
                # Update frame queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Process frame asynchronously if callback provided
                if self.frame_callback:
                    self.executor.submit(self._process_frame_async, frame)
                
                self.frame_count += 1
                self.last_processed_time = current_time
                
            except Exception as e:
                print(f"Error in streaming loop: {str(e)}")
                time.sleep(0.01)
    
    def _process_frame_async(self, frame):
        """Process frame asynchronously to avoid blocking the main stream"""
        try:
            if self.frame_callback:
                self.frame_callback(frame)
        except Exception as e:
            print(f"Error in async frame processing: {str(e)}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame without blocking"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except queue.Empty:
            return None
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get the most recent processed frame"""
        try:
            if not self.processed_frame_queue.empty():
                return self.processed_frame_queue.get_nowait()
            return None
        except queue.Empty:
            return None
    
    def set_performance_mode(self, mode: str):
        """Change performance mode on the fly"""
        if mode in ["fast", "balanced", "quality"]:
            self.performance_mode = mode
            self._setup_performance_mode()
            print(f"Performance mode changed to: {mode}")
        else:
            print(f"Invalid performance mode: {mode}. Use 'fast', 'balanced', or 'quality'")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'performance_mode': self.performance_mode,
            'resolution': f"{self.width}x{self.height}",
            'target_fps': self.fps,
            'frame_skip': self.frame_skip,
            'processing_interval': self.min_processing_interval,
            'queue_size': self.frame_queue.qsize(),
            'is_streaming': self.is_streaming
        }
    
    def start_recording(self, filename: str = None) -> bool:
        """Start video recording with performance optimizations"""
        if self.is_recording:
            print("Recording already active")
            return False
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"workout_recording_{timestamp}.mp4"
        
        output_path = self.output_dir / filename
        
        # Initialize video writer with performance codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print("Failed to initialize video writer")
            return False
        
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"Recording started: {filename}")
        return True
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            print("No recording active")
            return
        
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            print("Recording stopped")
    
    def _recording_loop(self):
        """Optimized recording loop"""
        while self.is_recording:
            try:
                frame = self.capture_frame()
                if frame is not None:
                    self.video_writer.write(frame)
                
                # Control recording frame rate
                time.sleep(1.0 / self.fps)
            except Exception as e:
                print(f"Error in recording loop: {str(e)}")
                time.sleep(0.01)
    
    def take_photo(self, filename: str = None) -> Optional[str]:
        """Take a single photo"""
        if not self.camera or not self.camera.isOpened():
            print("Camera not initialized")
            return None
        
        frame = self.capture_frame()
        if frame is None:
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
        
        output_path = self.output_dir / filename
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save photo
        success = cv2.imwrite(str(output_path), frame)
        if success:
            print(f"Photo saved: {filename}")
            return str(output_path)
        else:
            print("Failed to save photo")
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and capabilities"""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        info = {
            'camera_index': self.camera_index,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
            'hue': self.camera.get(cv2.CAP_PROP_HUE),
            'gain': self.camera.get(cv2.CAP_PROP_GAIN),
            'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
            'performance_mode': self.performance_mode,
            'frame_skip': self.frame_skip,
            'processing_interval': self.min_processing_interval
        }
        
        return info
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set camera property"""
        if not self.camera or not self.camera.isOpened():
            return False
        
        return self.camera.set(property_id, value)
    
    def get_camera_property(self, property_id: int) -> float:
        """Get camera property value"""
        if not self.camera or not self.camera.isOpened():
            return 0.0
        
        return self.camera.get(property_id)
    
    def list_available_cameras(self) -> list:
        """List available camera devices"""
        available_cameras = []
        
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        return available_cameras
    
    def switch_camera(self, new_camera_index: int) -> bool:
        """Switch to a different camera"""
        if self.is_streaming or self.is_recording:
            print("Cannot switch camera while streaming or recording")
            return False
        
        # Release current camera
        self.release_camera()
        
        # Initialize new camera
        self.camera_index = new_camera_index
        return self.initialize_camera()
    
    def apply_filters(self, frame: np.ndarray, filter_type: str = 'none') -> np.ndarray:
        """Apply various filters to the frame"""
        if filter_type == 'none':
            return frame
        elif filter_type == 'grayscale':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_type == 'blur':
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_type == 'edge_detection':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 100, 200)
        elif filter_type == 'sepia':
            # Apply sepia filter
            frame_float = frame.astype(np.float32)
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            sepia_frame = cv2.transform(frame_float, sepia_matrix)
            sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
            return sepia_frame
        else:
            print(f"Unknown filter type: {filter_type}")
            return frame
    
    def draw_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw text overlay on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # White
        thickness = 2
        
        # Add black background for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + baseline + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
    
    def __enter__(self):
        """Context manager entry"""
        if self.initialize_camera():
            return self
        else:
            raise RuntimeError("Failed to initialize camera")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release_camera()

# Keep the original CameraService for backward compatibility
class CameraService(OptimizedCameraService):
    """Legacy camera service - now inherits from OptimizedCameraService"""
    pass

if __name__ == "__main__":
    # Example usage
    with CameraService() as camera:
        # Take a photo
        photo_path = camera.take_photo()
        if photo_path:
            print(f"Photo taken: {photo_path}")
        
        # Get camera info
        info = camera.get_camera_info()
        print("Camera info:", info)
        
        # List available cameras
        available = camera.list_available_cameras()
        print(f"Available cameras: {available}")
