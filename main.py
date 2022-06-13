import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import warnings

from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample
# from utils import hamilton_detector, christov_detector, findpeaks, engzee_detector
# from ecg_detectors.ecgdetectors import Detectors, MWA, panPeakDetect, searchBack

np.random.seed(256)
sns.set()

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    data = None
    fs = 200
    lowcut = 0.05 * 3.3  # 9.9 beats per minute
    highcut = 15  # 900 beats per minute
    

# Sample rate and Desired Cutoff Frequencies


