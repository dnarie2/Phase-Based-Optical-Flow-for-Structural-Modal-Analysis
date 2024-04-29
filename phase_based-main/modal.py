import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to extract motion data from optical flow
def extract_motion_data(video_path, region_of_interest):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    motion_data = []
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        x, y = region_of_interest
        velocity = flow[y, x]  # Assuming (x, y) is the point of interest
        motion_data.append(velocity)

        prvs = next

    cap.release()
    return np.array(motion_data)

# Function to calculate Fourier transform
# Function to calculate Fourier transform with high-pass filter
def calculate_fourier_transform(motion_data, fps, cutoff_freq=0.1):
    # Apply high-pass filter to remove DC component
    motion_data -= np.mean(motion_data, axis=0)

    # Calculate Fourier transform
    displacement_data = np.cumsum(motion_data, axis=0) / fps
    spectrum = np.abs(np.fft.fft(displacement_data[:, 0]))  # Assuming only analyzing x-component
    frequencies = np.fft.fftfreq(len(displacement_data), d=1/fps)

    # Apply high-pass filter to remove low frequencies
    spectrum[frequencies < cutoff_freq] = 0
    return frequencies, spectrum


# Main function
if __name__ == "__main__":
    video_path = 'magnified_video.avi'
    fps = 30  # Update with actual frame rate of the video

    # Example region of interest (x, y) pixel coordinates
    region_of_interest = (100, 100)

    # Step 1: Extract motion data
    motion_data = extract_motion_data(video_path, region_of_interest)

    # Step 2: Calculate Fourier transform
    frequencies, spectrum = calculate_fourier_transform(motion_data, fps)

    # Plot Fourier spectrum
    plt.plot(frequencies, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum')
    plt.grid(True)
    plt.show()
