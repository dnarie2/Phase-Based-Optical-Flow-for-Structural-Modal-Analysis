import cv2
import numpy as np

# Load the video
video_path = 'Deployed_Magnified.mp4'
cap = cv2.VideoCapture(video_path)

# Get the dimensions of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window to display the video
cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)  # Enable resizing
cv2.resizeWindow('Optical Flow', frame_width, frame_height)  # Set window size

# Parameters for Lucas-Kanade method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Color for displaying the optical flow
color = (255, 0, 0)

# Take the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error reading video file")
    exit()

# Convert frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

# Select points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.9, minDistance=9, blockSize=9)

# Initialize list to store marker positions
marker_positions = [[] for _ in range(len(p0))]  # Each marker has its own list to store its positions

frame_count = 0
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

    # Draw the tracks and update marker positions
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int) 
        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        frame = cv2.circle(frame, (a, b), 5, color, -1)
        # Store the position of each marker at each frame
        marker_positions[i].append((a, b))

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Increment frame count
    frame_count += 1

    # Display the optical flow
    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', img)

    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Output marker positions to a CSV file
output_file = 'marker_positions.csv'
with open(output_file, 'w') as f:
    f.write('Frame,')
    for i in range(len(marker_positions)):
        f.write(f'Marker_{i+1}_X,Marker_{i+1}_Y,')
    f.write('\n')
    for frame_num, positions in enumerate(zip(*marker_positions), 1):
        f.write(f'{frame_num},')
        for pos in positions:
            f.write(f'{pos[0]},{pos[1]},')
        f.write('\n')

print(f'Marker positions saved to {output_file}')
