import cv2
import numpy as np

# Load the video
video_path = 'deployed.mp4'
output_video_path = 'deployed_filtered.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to highlight bright regions
    _, bright_regions = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Convert the binary image to BGR format
    bright_regions_bgr = cv2.cvtColor(bright_regions, cv2.COLOR_GRAY2BGR)

    # Apply bitwise AND to keep only the bright white regions
    filtered_frame = cv2.bitwise_and(frame, bright_regions_bgr)

    # Write the filtered frame to the output video
    out.write(filtered_frame)

    # Display the filtered frame
    cv2.imshow('Filtered Frame', filtered_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
