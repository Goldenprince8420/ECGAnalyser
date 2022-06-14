import numpy as np
from detector_utils import *
from utils import *


def pan_tompkins_detector(raw_ecg, mwa, fs, N):
    N = int(N / 100 * fs)
    mwa_peaks = panPeakDetect(mwa, fs)
    r_peaks = searchBack(mwa_peaks, raw_ecg, N)
    return r_peaks


def get_R_peaks(data, y, fs):
    # Derivative - provides QRS slope information.
    differentiated_ecg_measurements = np.ediff1d(y)

    # Squaring - intensifies values received in derivative.
    # This helps restrict false positives caused by T waves with higher than usual spectral energies..
    squared_ecg_measurements = differentiated_ecg_measurements ** 2

    # Moving-window integration.
    integration_window = 50  # Change proportionally when adjusting frequency (in samples)
    integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))

    # Fiducial mark - peak detection on integrated measurements.
    rpeaks = pan_tompkins_detector(data, integrated_ecg_measurements, fs, integration_window)
    return differentiated_ecg_measurements, squared_ecg_measurements, integrated_ecg_measurements, rpeaks


def get_other_detectors(data, fs, y):
    differentiated_ecg_measurements, squared_ecg_measurements, integrated_ecg_measurements, rpeaks = get_R_peaks(data,
                                                                                                                 y, fs)
    hamilton_rpeaks = hamilton_detector(data, fs, y)
    christov_rpeaks = christov_detector(fs, data)
    detected_peaks_indices = findpeaks(data=integrated_ecg_measurements, limit=0.35, spacing=100)
    engzee_rpeaks = engzee_detector(y, fs, data)
    # detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

    # https://github.com/luishowell/ecg-detectors
    detectors = Detectors(fs)

    detectors_two_average_rpeaks = detectors.two_average_detector(data)
    # detectors_matched_filter_rpeaks = detectors.matched_filter_detector(data)
    detectors_swt_rpeaks = detectors.swt_detector(data)
    detectors_christov_rpeaks = detectors.christov_detector(data)
    detectors_hamilton_rpeaks = detectors.hamilton_detector(data)
    detector_engzee_rpeaks = detectors.engzee_detector(data)
    return detectors_swt_rpeaks, detectors_christov_rpeaks, detectors_hamilton_rpeaks, detector_engzee_rpeaks, \
           [detectors_two_average_rpeaks, hamilton_rpeaks, christov_rpeaks, detected_peaks_indices, engzee_rpeaks]


def hrv_features(rpeaks, fs):
    rr = np.diff(rpeaks) / fs * 1000  # in miliseconds
    hr = 60 * 1000 / rr
    print("rr =", rr)
    print("hr =", hr)

    # RMSSD ("root mean square of successive differences"), the square root of the mean of the squares of the successive
    # differences between adjacent NNs.
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    print("RMSSD =", round(rmssd, 3))

    # Mean RR
    mean_rr = np.mean(rr)

    # SDNN, the standard deviation of NN intervals. Often calculated over a 24-hour period.
    # SDANN, the standard deviation of the average NN intervals calculated over short periods, usually 5 minutes.
    # SDNN is therefore a measure of changes in heart rate due to cycles longer than 5 minutes.
    # SDNN reflects all the cyclic components responsible for variability in the period of recording,
    # therefore it represents total variability.
    sdnn = np.std(rr)
    # print("SDNN =", round(sdnn, 3), "\n")

    print("Mean RR = {} ± {}".format(round(mean_rr, 3), round(sdnn, 3)))

    # Mean HR
    mean_hr = np.mean(hr)

    # STD HR
    std_hr = np.std(hr)
    print("\nMean HR =", round(mean_hr, 3), "±", round(std_hr, 3))
    # print("std_hr =", round(std_hr, 3))

    # Min HR
    min_hr = np.min(hr)
    print("min_hr =", round(min_hr, 3))

    # Max HR
    max_hr = np.max(hr)
    print("max_hr =", round(max_hr, 3), "\n")

    # NNxx: sum absolute differences that are larger than 50ms
    nnxx = np.sum(np.abs(np.diff(rr)) > 50) * 1
    print("NNxx =", nnxx)

    # pNNx: fraction of nnxx of all rr-intervals
    pnnx = 100 * nnxx / len(rr)
    print("pNNx =", round(pnnx, 3))
