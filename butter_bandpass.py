from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample


class ButterBandpass:
    def __init__(self, lowcut, highcut, fs, order):
        self.low = lowcut
        self.high = highcut
        self.fs = fs
        self.order = order
        self.sos = None
        self.zi = None
        self.z = None
        self.z2 = None

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.low / nyq
        high = self.high / nyq
        self.sos = butter(self.order, [low, high],
                          analog = False,
                          btype = "band",
                          output = "sos")

    def butter_bandpass_filter(self, data):
        self.butter_bandpass()
        y = sosfilt(self.sos, data)
        return y

    def butter_bandpass_filter_once(self, data):
        self.butter_bandpass()
        self.zi = sosfilt_zi(self.sos)
        self.z, _ = sosfilt(self.sos, data, zi = self.zi * data[0])

    def butter_bandpass_filter_again(self):
        self.butter_bandpass_filter_once()
        self.z2, _ = sosfilt(self.sos, self.z, zi = self.zi * self.z[0])

    def butter_bandpass_for_back_filter(self, data):
        self.butter_bandpass()
        y = sosfiltfilt(self.sos, data)
        return y


