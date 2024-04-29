import cv2
import numpy as np

# Step 1: Motion Magnification
def magnify_motion(video_path, alpha=30, freq_min=20, freq_max=90):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('magnified_video.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = np.float32(gray_frame) / 255.0
        
        # Apply temporal filter
        filtered_frame = cv2.bilateralFilter(gray_frame, 5, 50, 50)
        
        # Magnify motion
        magnified_frame = filtered_frame + alpha * (gray_frame - filtered_frame)
        
        # Clip values
        magnified_frame = np.clip(magnified_frame, 0, 1)
        
        # Convert back to uint8
        magnified_frame = np.uint8(magnified_frame * 255)
        
        out.write(cv2.cvtColor(magnified_frame, cv2.COLOR_GRAY2BGR))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Step 2: Optical Flow
def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Visualize optical flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('Optical Flow', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    video_path = 'magnified_video.avi'
    
    # Step 1: Magnify motion
    magnify_motion('Videos/deployed_og.mp4') # Replace 'your_video_path.mp4' with your actual video path
    
    # Step 2: Optical flow
    calculate_optical_flow(video_path)
