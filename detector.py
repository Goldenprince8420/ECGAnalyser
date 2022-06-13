def pan_tompkins_detector(raw_ecg, mwa, fs, N):
    N = int(N / 100 * fs)
    mwa_peaks = panPeakDeteck(mwa, fs)
    r_peaks = searchBack(mwa_peaks, raw_ecg, N)
    return r_peaks