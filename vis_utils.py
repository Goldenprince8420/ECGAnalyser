import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from butter_bandpass import *
from detector import *


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


def plot_R_peaks(data, low, high, fs, order):
    x = data[:300]
    bandpass = ButterBandpass(low, high, fs, order)
    y = bandpass.butter_bandpass_for_back_filter(x)
    differentiated_ecg_measurements, squared_ecg_measurements, integrated_ecg_measurements, rpeaks = get_R_peaks(y, data, fs)
    figsize = (20, 10)

    plt.figure(figsize=figsize)

    plt.subplot2grid((6, 2), (0, 0))
    # plt.subplot(621)
    # plt.figure(1, figsize=left_figsize)
    # plt.clf()
    plt.xticks([])
    plt.plot(data)
    plt.title("Raw ecg data")
    plt.grid(True)
    # plt.show()

    # plt.figure(figsize=left_figsize)
    plt.subplot2grid((6, 2), (1, 0))
    # plt.subplot(622)
    # plt.clf()
    plt.xticks([])
    plt.plot(y)
    plt.title("Filtered ecg data")
    plt.grid(True)
    # plt.show()

    # plt.figure(figsize=left_figsize)
    plt.subplot2grid((6, 2), (2, 0))
    # plt.subplot(623)
    # plt.clf()
    plt.xticks([])
    plt.plot(differentiated_ecg_measurements)
    plt.title("Differentiated ecg data")
    plt.grid(True)
    # plt.show()

    # plt.figure(figsize=left_figsize)
    plt.subplot2grid((6, 2), (3, 0))
    # plt.subplot(624)
    # plt.clf()
    plt.xticks([])
    plt.plot(squared_ecg_measurements)
    plt.title("Squared ecg data")
    plt.grid(True)
    # plt.show()

    # plt.figure(figsize=left_figsize)
    plt.subplot2grid((6, 2), (4, 0))
    # plt.subplot(625)
    # plt.clf()
    plt.xticks([])
    plt.plot(integrated_ecg_measurements)
    plt.title("Integrated ECG data")
    plt.grid(True)
    # plt.show()

    ymin = np.min(data)
    ymax = np.max(data)
    alpha = 0.2 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    # plt.figure(figsize=left_figsize)
    # plt.subplot(626)
    plt.subplot2grid((6, 2), (5, 0))
    # plt.clf()
    plt.ylabel("Amplitude (dB)")
    # Calculate time values in seconds
    times = np.arange(data.shape[0], dtype='float') / fs
    plt.xlabel("Time (s)")
    plt.plot(times, data)
    plt.vlines(rpeaks / fs, ymin, ymax,
               color="r",
               linewidth=2,
               label="R-peaks")
    plt.title("Raw ecg data with pulse stream")
    plt.grid(True)
    # plt.show()

    plt.subplot2grid((6, 2), (0, 1))
    pan_tompkins_fig1_a = mpimg.imread("images/a_original_signal.png")
    plt.title("Original signal - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_a)

    plt.subplot2grid((6, 2), (1, 1))
    pan_tompkins_fig1_b = mpimg.imread("images/b_output_of_bandpass_filter.png")
    plt.title("Output of bandpass filter - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_b)

    plt.subplot2grid((6, 2), (2, 1))
    pan_tompkins_fig1_c = mpimg.imread("images/c_ouput_of_differentiator.png")
    plt.title("Output of differentiator - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_c)

    plt.subplot2grid((6, 2), (3, 1))
    pan_tompkins_fig1_d = mpimg.imread("images/d_output_of_squaring_process.png")
    plt.title("Output of Squaring process - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_d)

    plt.subplot2grid((6, 2), (4, 1))
    pan_tompkins_fig1_e = mpimg.imread("images/e_results_of_moving_window_integration.png")
    plt.title("Results of moving window integration - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_e)

    plt.subplot2grid((6, 2), (5, 1))
    pan_tompkins_fig1_g = mpimg.imread("images/g_output_pulse_stream.png")
    plt.title("Output pulse stream - from Pan, Tompkins et al.")
    plt.axis("off")
    plt.imshow(pan_tompkins_fig1_g)


def plot_detector_comparison(data, fs, y):
    differentiated_ecg_measurements, squared_ecg_measurements, integrated_ecg_measurements, rpeaks = \
        get_R_peaks(data, y, fs)
    peaks, detectors_swt_rpeaks, detectors_christov_rpeaks, \
        detectors_hamilton_rpeaks, detector_engzee_rpeaks, peaks  = \
        get_other_detectors(data, fs, y)
    detectors_two_average_rpeaks, hamilton_rpeaks, christov_rpeaks, detected_peaks_indices, engzee_rpeaks = peaks
    figsize = (20, 10)

    plt.figure(figsize=figsize)

    ymin = np.min(data)
    ymax = np.max(data)
    alpha = 0.2 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    grid = (8, 2)

    index = 0

    plt.subplot2grid(grid, (index, 0), colspan=2)
    plt.xticks([])
    plt.plot(data)
    plt.vlines(rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Pan Tomkins with custom filter")
    plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0), colspan=2)
    plt.xticks([])
    plt.plot(data)
    plt.vlines(detected_peaks_indices, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Janko Slavic with custom filter")
    plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0), colspan=2)
    plt.xticks([])
    plt.plot(data)
    plt.vlines(detectors_two_average_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Two average with default filter")
    plt.grid(True)

    # index += 1

    # plt.subplot2grid(grid, (index, 0), colspan = 2)
    # plt.xticks([])
    # plt.plot(data)
    # plt.vlines(detectors_matched_filter_rpeaks, ymin, ymax,
    #                color="r",
    #                linewidth=2)
    # plt.title("R-peaks by Matched filter detector with default filter")
    # plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0), colspan=2)
    plt.xticks([])
    plt.plot(data)
    plt.vlines(detectors_swt_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by SWT detector with default filter")
    plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0))
    plt.plot(data)
    plt.xticks([])
    plt.vlines(hamilton_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Hamilton with custom filter")
    plt.grid(True)

    plt.subplot2grid(grid, (index, 1))
    plt.plot(data)
    plt.xticks([])
    plt.vlines(detectors_hamilton_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Hamilton with default filter")
    plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0))
    plt.plot(data)
    plt.xticks([])
    plt.vlines(engzee_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Engzee with custom filter")
    plt.grid(True)

    plt.subplot2grid(grid, (index, 1))
    plt.plot(data)
    plt.xticks([])
    plt.vlines(detector_engzee_rpeaks, ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Engzee with default filter")
    plt.grid(True)

    index += 1

    plt.subplot2grid(grid, (index, 0))
    # Calculate time values in seconds
    times = np.arange(data.shape[0], dtype='float') / fs
    plt.xlabel("Time (s)")
    plt.plot(times, data)
    plt.vlines([r / fs for r in christov_rpeaks], ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Christov with custom filter")
    plt.grid(True)

    plt.subplot2grid(grid, (index, 1))
    # Calculate time values in seconds
    times = np.arange(data.shape[0], dtype='float') / fs
    plt.xlabel("Time (s)")
    plt.plot(times, data)
    plt.vlines([r / fs for r in detectors_christov_rpeaks], ymin, ymax,
               color="r",
               linewidth=2)
    plt.title("R-peaks by Christov with default filter")
    plt.grid(True)

