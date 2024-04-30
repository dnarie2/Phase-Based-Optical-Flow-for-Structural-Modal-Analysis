import cv2
import numpy as np

# Load the video
video_path = 'IROSA.mp4'
cap = cv2.VideoCapture(video_path)

# Parameters for Lucas-Kanade method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Color for displaying the optical flow
color = (0,255, 0)

# Take the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error reading video file")
    exit()

# Convert frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

# Select a point to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.9, minDistance=9, blockSize=9)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        frame = cv2.circle(frame, (a, b), 5, color, -1)

    # Display the optical flow
    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', img)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
