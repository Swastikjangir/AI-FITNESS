"""
Camera service module for AI Fitness Coach.

This module provides camera functionality for real-time video streaming
with performance optimizations.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor

from ai_fitness.config.settings import get_settings

class OptimizedCameraService:
    """Optimized camera service for smooth real-time video processing"""
    
    def __init__(self, camera_index: int = None, performance_mode: str = "balanced"):
        self.settings = get_settings()
        self.camera_index = camera_index or self.settings.camera_index
        self.camera = None
        self.is_streaming = False
        self.frame_callback = None
        self.streaming_thread = None
        
        # Performance optimization settings
        self.performance_mode = performance_mode
        self._setup_performance_mode()
        
        # Frame processing optimization
        self.frame_skip = 2  # Process every Nth frame
        self.frame_count = 0
        self.last_processed_time = 0
        self.min_processing_interval = 1.0 / 15  # Max 15 FPS processing
        
        # Threading and queuing for smooth streaming
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer for smooth streaming
        self.latest_frame = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Camera properties
        self.width = self.settings.camera_width
        self.height = self.settings.camera_height
        self.fps = self.settings.camera_fps
        
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
            print("Camera not initialized or not opened")
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Resize frame if needed for performance
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                return frame
            else:
                print("Failed to capture frame from camera")
                return None
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
    
    def start_streaming(self, callback: Callable[[np.ndarray], None] = None):
        """Start optimized video streaming"""
        if self.is_streaming:
            print("Streaming already active")
            return
        
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                print("Failed to initialize camera for streaming")
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
        print("Starting optimized stream loop")
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
                
                # Update latest frame (non-blocking)
                self.latest_frame = frame.copy()
                
                # Update frame queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
                
                # Process frame asynchronously if callback provided
                if self.frame_callback:
                    self.executor.submit(self._process_frame_async, frame)
                
                self.frame_count += 1
                self.last_processed_time = current_time
                
                # Debug: print frame count every 30 frames
                if self.frame_count % 30 == 0:
                    print(f"Streaming: processed {self.frame_count} frames")
                
            except Exception as e:
                print(f"Error in streaming loop: {str(e)}")
                time.sleep(0.01)
        
        print("Streaming loop ended")
    
    def _process_frame_async(self, frame):
        """Process frame asynchronously to avoid blocking the main stream"""
        try:
            if self.frame_callback:
                self.frame_callback(frame)
        except Exception as e:
            print(f"Error in async frame processing: {str(e)}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame without blocking"""
        if self.latest_frame is not None:
            return self.latest_frame
        else:
            print("No latest frame available")
            return None
    
    def is_camera_available(self) -> bool:
        """Check if camera is available and working"""
        return self.camera is not None and self.camera.isOpened()
    
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
            'is_streaming': self.is_streaming,
            'camera_available': self.is_camera_available()
        }
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and capabilities"""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        info = {
            'camera_index': self.camera_index,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'performance_mode': self.performance_mode,
            'frame_skip': self.frame_skip,
            'processing_interval': self.min_processing_interval
        }
        
        return info
    
    def stream_frames_streamlit(self, st_placeholder, stop_flag=None, max_fps=30):
        """
        Stream video frames to Streamlit placeholder in real-time.
        
        Args:
            st_placeholder: Streamlit placeholder for displaying images
            stop_flag: Optional threading.Event() to signal stopping
            max_fps: Maximum frames per second (default: 30)
        """
        if not self.camera or not self.camera.isOpened():
            print("Camera not initialized for streaming")
            return False
        
        # Calculate delay based on max_fps
        frame_delay = 1.0 / max_fps
        
        # Initialize camera if not already streaming
        if not self.is_streaming:
            self.start_streaming()
        
        print(f"Starting Streamlit video stream with {max_fps} FPS")
        
        try:
            while True:
                # Check stop flag
                if stop_flag and stop_flag.is_set():
                    print("Streamlit streaming stopped by flag")
                    break
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display in Streamlit placeholder
                st_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay to prevent CPU overload
                time.sleep(frame_delay)
                
        except Exception as e:
            print(f"Error in Streamlit streaming: {str(e)}")
            return False
        
        print("Streamlit video stream ended")
        return True
    
    def stream_frames_streamlit_with_processing(self, st_placeholder, processing_callback=None, stop_flag=None, max_fps=30):
        """
        Stream video frames to Streamlit with optional processing callback.
        
        Args:
            st_placeholder: Streamlit placeholder for displaying images
            processing_callback: Optional callback function(frame) -> processed_frame
            stop_flag: Optional threading.Event() to signal stopping
            max_fps: Maximum frames per second (default: 30)
        """
        if not self.camera or not self.camera.isOpened():
            print("Camera not initialized for streaming")
            return False
        
        # Calculate delay based on max_fps
        frame_delay = 1.0 / max_fps
        
        # Initialize camera if not already streaming
        if not self.is_streaming:
            self.start_streaming()
        
        print(f"Starting Streamlit video stream with processing at {max_fps} FPS")
        
        try:
            while True:
                # Check stop flag
                if stop_flag and stop_flag.is_set():
                    print("Streamlit streaming stopped by flag")
                    break
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Apply processing callback if provided
                if processing_callback:
                    try:
                        processed_frame = processing_callback(frame)
                        if processed_frame is not None:
                            frame = processed_frame
                    except Exception as e:
                        print(f"Error in processing callback: {str(e)}")
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display in Streamlit placeholder
                st_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay to prevent CPU overload
                time.sleep(frame_delay)
                
        except Exception as e:
            print(f"Error in Streamlit streaming with processing: {str(e)}")
            return False
        
        print("Streamlit video stream with processing ended")
        return True
    
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
        # Get camera info
        info = camera.get_camera_info()
        print("Camera info:", info)
