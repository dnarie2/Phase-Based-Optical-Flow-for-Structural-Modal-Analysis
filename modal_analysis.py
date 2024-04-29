import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data from Excel sheet
excel_file = "marker_positions_1.xlsx" 
df = pd.read_excel(excel_file)

frames = df["Frame"].values
y_displacement = df["Marker_Y"].values
x_displacement = df["Marker_X"].values

# Calculate marker displacement
initial_x = x_displacement[0]
initial_y = y_displacement[0]
marker_displacement = np.sqrt((x_displacement - initial_x)**2 + (y_displacement - initial_y)**2)

# Normalize marker displacement
normalized_marker_displacement = (marker_displacement - np.mean(marker_displacement)) / np.std(marker_displacement)

# Apply windowing function to reduce spectral leakage
window = np.hamming(len(normalized_marker_displacement))
windowed_marker_displacement = normalized_marker_displacement * window

# Perform Fourier Transform
displacement_fft = np.fft.fft(windowed_marker_displacement)
frequencies = np.fft.fftfreq(len(normalized_marker_displacement), 1 / (frames[1] - frames[0]))

# Find peaks in the spectrum
peaks, _ = find_peaks(np.abs(displacement_fft), height=0)

# Back-transform peaks to time domain to obtain ODS
ODS = np.fft.ifft(np.eye(len(normalized_marker_displacement))[:, peaks])

# Calculate phases of identified peaks
phases = np.angle(displacement_fft[peaks])

# Plot ODS
plt.figure(figsize=(10, 5))
plt.plot(frames, np.real(ODS))
plt.xlabel('Frames')
plt.ylabel('Normalized ODS')
plt.title('Normalized Operational Deflection Shapes')
plt.grid(True)
plt.show()

# Plot phases
plt.figure(figsize=(10, 5))
plt.plot(frequencies[peaks], phases, 'o')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase')
plt.title('Phase of Identified Peaks')
plt.grid(True)
plt.show()
