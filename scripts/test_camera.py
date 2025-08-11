import cv2
import numpy as np

def test_camera():
    """Simple camera test to verify functionality"""
    print("Testing camera...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("Please check:")
        print("1. Webcam is connected")
        print("2. No other application is using the camera")
        print("3. Camera permissions are granted")
        return False
    
    print("✅ Camera opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from camera!")
        cap.release()
        return False
    
    print("✅ Frame read successfully!")
    print(f"Frame shape: {frame.shape}")
    
    # Show the frame for 3 seconds
    print("Showing camera feed for 3 seconds...")
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text to frame
        cv2.putText(frame, "Camera Test - Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera Test', frame)
        
        # Check if 3 seconds have passed or 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
        if elapsed_time > 3:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Camera test completed successfully!")
    return True

if __name__ == "__main__":
    test_camera() 