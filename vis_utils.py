import numpy as np
import matplotlib.pyplot as plt
from butter_bandpass import *


def plot_raw_ecg(data, fs):
    plt.figure(figsize = (20, 5))
    # Time value in seconds
    times = np.arange(data.shape[0], dtype = "float") / fs
    plt.plot(times, data)
    plt.xlabel("Time(s)")
    plt.ylabel("ECG sample raw data")
    plt.show()


def plot_frequency_response(low, high, fs):
    plt.figure(1, figsize = (20, 8))
    plt.clf()
    for order in [1, 2, 4, 6, 8]:
        bandpass = ButterBandpass(low, high, fs, order)
        bandpass.butter_bandpass()
        w, h = sosfreqz(bandpass.sos, worN = 2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label = "order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5)],
             "--", labels = "sqrt(0.5)")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Gain")
    plt.title("Frequency response for a few different orders")
    plt.grid(True)
    plt.legend(loc="best")


def plot_filtered_noise_signal(x, low, high, fs, over):
    times = np.arange(x.shape[0], dtype = "float") / fs
    bandpass = ButterBandpass(low, high, fs, over)
    y1 = bandpass.butter_bandpass_filter(x)
    y2 = bandpass.butter_bandpass_for_back_filter(x)
    plt.figure(2, figsize = (20, 8))
    plt.clf()
    plt.ylabel("Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.plot(times, x, "g", linewidth = 0.5, label = "Noisy Signal")
    plt.legend(loc = "lower left")
    plt.twinx()
    plt.plot(times, y1,
             label = "Filtered Signal (%g Hz) using sosfilt(without initial conditions)" % fs)
    plt.plot(times, y2,
             label = "Filtered Signal using softfiltfilt " % fs)
    plt.grid(True)
    plt.axis("tight")
    plt.legend(loc = "upper right")
    plt.show()


def plot_phase_shift(data, low, high, fs, order):
    x = data[:300]
    bandpass = ButterBandpass(low, high, fs, order)
    y1 = bandpass.butter_bandpass_filter(x)
    y2 = bandpass.butter_bandpass_for_back_filter(x)
    bandpass.butter_bandpass_filter_once(x)
    bandpass.butter_bandpass_filter_again(x)
    times = np.arange(x.shape[0], dtype = "float") / fs

    plt.figure(figsize = (20, 8))
    plt.ylabel("Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.plot(times, x, "g",
             linewidth = 0.5, label = "Noisy Signal")
    plt.legend(loc = "lower left")
    plt.twinx()
    plt.plot(times, y1,
             label = "Filtered Signal (%g Hz) using sosfilt(Without Initial Condition)" % fs)
    plt.plot(times, bandpass.z, "r--",
             label = "Filtered signal (%g Hz) using sosfilt and sosfilt_zi (with initial conditions), once" % fs)
    plt.plot(times, bandpass.z2, "r",
             label = "Filtered signal (%g Hz) using sosfilt and sosfilt_zi (with initial conditions), twice" % fs)
    plt.grid(True)
    plt.axis("tight")
    plt.legend(loc = "upper right")
    plt.show()








