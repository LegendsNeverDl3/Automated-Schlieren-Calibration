import cv2
import numpy as np
import sys
from collections import deque

def calculate_scores(frame, history_frames):
    """
    Calculates two quality scores for a Schlieren frame.
    1. Coarse Score (General Variance): Detects heat plumes (indicates we are close).
    2. Fine Score (Acoustic Waves): Uses Temporal Averaging to blur out heat,
       then applies a Sobel filter to detect horizontal standing waves.
    """
    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------------------------------------
    # 1. COARSE SCORE (General Variance - Heat + Everything)
    # ---------------------------------------------------------
    coarse_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # ---------------------------------------------------------
    # 2. FINE SCORE (Acoustic Standing Waves)
    # ---------------------------------------------------------
    # Maintain a rolling history of the last 10 frames
    history_frames.append(gray)
    if len(history_frames) > 10:
        history_frames.popleft()
        
    # Temporal Averaging: Blurs out moving heat, leaves stationary sound waves
    avg_gray = np.mean(history_frames, axis=0).astype(np.uint8)
    
    # Since the waves are vertical C-shapes, we use a Laplacian filter
    # to catch the sharp edges in all directions on the stationary (averaged) image!
    edges = cv2.Laplacian(avg_gray, cv2.CV_64F)
    fine_score = np.var(edges)
    
    # Calculate Mean Brightness
    mean_brightness = cv2.mean(gray)[0] / 255.0
    
    return mean_brightness, coarse_score, fine_score, avg_gray

def process_video(video_source):
    print(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    cv2.namedWindow('Schlieren Analysis', cv2.WINDOW_NORMAL)
    history_frames = deque()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        brightness, coarse, fine, avg_frame = calculate_scores(frame, history_frames)
        
        # Display the scores on the frame
        cv2.putText(frame, f"Brightness: {brightness*100:.1f}%", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"COARSE (Heat): {coarse:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2) # Orange
        cv2.putText(frame, f"FINE (Acoustic): {fine:.1f}", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # Blue
        
        # Show both the live feed and the "Filtered" feed side-by-side
        # (Convert avg_frame to BGR so we can stack them)
        avg_bgr = cv2.cvtColor(avg_frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(avg_bgr, "Filtered (Heat Blurred Out)", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        combined_display = np.hstack((frame, avg_bgr))
        cv2.imshow('Schlieren Analysis', combined_display)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python schlieren_cv_test.py <path_to_video_file>")
        print("Falling back to live webcam (index 0) for testing...")
        process_video(0)
    else:
        process_video(sys.argv[1])
