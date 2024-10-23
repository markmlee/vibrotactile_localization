import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set random seed for reproducibility
np.random.seed(42)

def generate_signals(fs=1000, t_delay=0.001, duration=0.1):
    """
    Generate two test signals:
    Signal 1: sum of 100Hz and 200Hz sinusoids
    Signal 2: same but delayed by t_delay and different amplitudes
    """
    t = np.arange(0, duration, 1/fs)
    
    # Original signal
    x = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t)
    
    # Delayed signal with different amplitudes
    # Use numpy roll for the delay
    n_delay = int(t_delay * fs)  # convert time delay to samples
    y = 2*np.sin(2*np.pi*100*t) + 3*np.sin(2*np.pi*200*t)
    y = np.roll(y, n_delay)
    
    return t, x, y

def gcc_phat(x, y, fs):
    """
    Compute GCC-PHAT for two signals
    
    Parameters:
    -----------
    x, y : numpy arrays
        Input signals
    fs : float
        Sampling frequency
    
    Returns:
    --------
    t : numpy array
        Time lag axis
    gcc : numpy array
        GCC-PHAT correlation
    """
    n = len(x)
    
    # Ensure signals are same length
    if len(x) != len(y):
        raise ValueError("Signals must have same length")
    
    # Apply Hanning window to reduce edge effects
    window = signal.windows.hann(n)
    x = x * window
    y = y * window

    print(f"dimension of x: {x.shape}, dimension of y: {y.shape}")
    
    # Compute FFT
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    print(f"dimension of X: {X.shape}, dimension of Y: {Y.shape}")
    
    # Cross-spectrum with small epsilon to avoid division by zero
    eps = 1e-10
    Sxy = X * np.conj(Y)
    
    # PHAT weighting
    Sxy_phat = Sxy / (np.abs(Sxy) + eps)

    print(f"dimension of Sxy_phat: {Sxy_phat.shape}")
    
    # Inverse FFT and normalization
    gcc = np.fft.irfft(Sxy_phat)
    print(f"dimension of gcc: {gcc.shape}")
    gcc = np.roll(gcc, len(gcc)//2)
    print(f"dimension of gcc: {gcc.shape}")
    
    # Create time axis
    t = np.linspace(-0.5*n/fs, 0.5*n/fs, n)

    
    return t, gcc

# Parameters
fs = 1000  # Sampling frequency (Hz)
t_delay = 0.001  # True time delay (1ms)
duration = 0.1  # Signal duration (seconds)

# Generate signals
t, x, y = generate_signals(fs, t_delay, duration)

# Compute GCC-PHAT
t_gcc, gcc = gcc_phat(x, y, fs)
print(f"dimension of t_gcc: {t_gcc.shape}, dimension of gcc: {gcc.shape}")

# Compute spectra for visualization
n = len(t)
X = np.fft.rfft(x) / n  # Normalize by signal length
Y = np.fft.rfft(y) / n
Sxy = X * np.conj(Y)
freqs = np.fft.rfftfreq(n, 1/fs)

# Create plots
plt.figure(figsize=(15, 12))

# Plot 1: Original signals
plt.subplot(4, 1, 1)
plt.plot(t*1000, x, label='Signal 1')
plt.plot(t*1000, y, label='Signal 2')
plt.title('Original Signals')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 2: Magnitude spectrum of signals
plt.subplot(4, 1, 2)
plt.plot(freqs, 20*np.log10(np.abs(X) + 1e-10), label='Signal 1')
plt.plot(freqs, 20*np.log10(np.abs(Y) + 1e-10), label='Signal 2')
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# Plot 3: Cross-spectrum phase
plt.subplot(4, 1, 3)
# phase = np.unwrap(np.angle(Sxy))  # Unwrap phase for better visualization
# plt.plot(freqs, phase)
# plt.title('Cross-Spectrum Phase (Unwrapped)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (radians)')
# plt.grid(True)

# Create two subplots side by side for cross-spectrum analysis
plt.subplot(4, 2, 5)
# Plot magnitude of cross-spectrum
plt.plot(freqs, np.abs(Sxy))
plt.title('Cross-Spectrum Magnitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
# Add vertical lines at our frequencies of interest
plt.axvline(x=100, color='r', linestyle='--', label='100 Hz')
plt.axvline(x=200, color='g', linestyle='--', label='200 Hz')
plt.legend()

plt.subplot(4, 2, 6)
# Plot phase of cross-spectrum
phase = np.unwrap(np.angle(Sxy))
plt.plot(freqs, phase)
plt.title('Cross-Spectrum Phase')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
# Add vertical lines at our frequencies of interest
plt.axvline(x=100, color='r', linestyle='--', label='100 Hz')
plt.axvline(x=200, color='g', linestyle='--', label='200 Hz')
plt.legend()


# Plot 4: GCC-PHAT result
plt.subplot(4, 1, 4)
plt.plot(t_gcc*1000, gcc)
plt.title('GCC-PHAT Result')
plt.xlabel('Time Lag (ms)')
plt.ylabel('Correlation')
plt.grid(True)

# Add vertical line at true delay
plt.axvline(x=t_delay*1000, color='r', linestyle='--', 
            label=f'True delay ({t_delay*1000:.1f} ms)')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()

# Analyze results
estimated_delay = t_gcc[np.argmax(gcc)]
print(f"True delay: {t_delay*1000:.3f} ms")
print(f"Estimated delay: {estimated_delay*1000:.3f} ms")
print(f"Error: {abs(estimated_delay - t_delay)*1000:.3f} ms")

# Show SNR
signal_power = np.mean(x**2)
noise_power = np.mean((y - x)**2)
snr = 10 * np.log10(signal_power/noise_power)
print(f"SNR: {snr:.1f} dB")