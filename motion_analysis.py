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

# Plot the time history of marker displacement
plt.figure(figsize=(10, 5))
plt.plot(frames, marker_displacement, label='Marker Displacement')
plt.xlabel('Frames')
plt.ylabel('Displacement')
plt.title('Time History of Marker Displacement')
plt.legend()
plt.grid(True)
plt.show()

# Perform Fourier Transform to identify dominant frequencies
Fs = frames[1] - frames[0]
N = len(marker_displacement)
frequencies = np.fft.fftfreq(N, 1/Fs)[:N//2]
marker_displacement_fft = np.fft.fft(marker_displacement)
amplitudes = np.abs(marker_displacement_fft)[:N//2]

# Plot Fourier amplitude spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, amplitudes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Amplitude Spectrum of Marker Displacement')
plt.grid(True)
plt.show()

# Find peaks in the spectrum to identify natural frequencies
peaks, _ = find_peaks(amplitudes, height=0)
natural_frequencies = frequencies[peaks]

print("Natural Frequencies (Hz):", natural_frequencies)
