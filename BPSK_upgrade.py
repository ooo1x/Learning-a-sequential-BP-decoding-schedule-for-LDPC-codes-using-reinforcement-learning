import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from math import pi, floor

def generate_binary_signal(size, t):
    a = np.random.randint(0, 2, size)
    m = np.zeros(len(t), dtype=np.float32)
    for i in range(len(t)):
        m[i] = a[floor(t[i])]
    return m

def modulate_signal(fc, ts, m):
    coherent_carrier = np.cos(2 * pi * fc * ts)
    bpsk = np.cos(2 * pi * fc * ts + pi * (m - 1) + pi / 4)
    return bpsk, coherent_carrier

def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y

def apply_filters(fs, noise_bpsk, coherent_carrier):
    b11, a11 = signal.ellip(5, 0.5, 60, [2000 * 2 / fs, 6000 * 2 / fs], btype='bandpass', analog=False, output='ba')
    b12, a12 = signal.ellip(5, 0.5, 60, 2000 * 2 / fs, btype='lowpass', analog=False, output='ba')
    bandpass_out = signal.filtfilt(b11, a11, noise_bpsk)
    coherent_demod = bandpass_out * (coherent_carrier * 2)
    lowpass_out = signal.filtfilt(b12, a12, coherent_demod)
    return lowpass_out

def detect_signal(lowpass_out, size):
    detection_bpsk = np.zeros(len(t), dtype=np.float32)
    flag = np.zeros(size, dtype=np.float32)
    for i in range(size):
        tempF = np.sum(lowpass_out[i * 100: (i + 1) * 100])
        flag[i] = 1 if tempF > 0 else 0
        detection_bpsk[i * 100: (i + 1) * 100] = flag[i]
    return detection_bpsk

# Parameters
size = 20  # Number of symbols
sampling_t = 0.01  # Sampling interval
fc = 4000  # Carrier frequency
fs = 40 * fc  # Sampling frequency

t = np.arange(0, size, sampling_t)
ts = np.arange(0, (100 * size) / fs, 1 / fs)
m = generate_binary_signal(size, t)
bpsk, coherent_carrier = modulate_signal(fc, ts, m)
noise_bpsk = awgn(bpsk, 5)
lowpass_out = apply_filters(fs, noise_bpsk, coherent_carrier)
detection_bpsk = detect_signal(lowpass_out, size)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(t, m)
axs[0].set_title('Generate N binary bits signal', fontsize=20)
axs[0].axis([0, size, -0.5, 1.5])
axs[1].plot(t, bpsk, 'r')
axs[1].set_title('BPSK modulated signal',  fontsize=20)
axs[1].axis([0, size, -1.5, 1.5])
axs[2].plot(t, noise_bpsk, 'r')
axs[2].set_title('BPSK modulated signal + Noise', fontsize=20)
axs[2].axis([0, size, -1.5, 1.5])
fig2, bx = plt.subplots(2, 1, figsize=(10, 15))
bx[0].plot(t, lowpass_out)
bx[0].set_title('After low pass filtered signal', fontsize=20)
bx[0].axis([0, size, -1.5, 1.5])
bx[1].plot(t, detection_bpsk)
bx[1].set_title('BPSK signal after sampling and decision', fontsize=20)
plt.show()