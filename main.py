import seaborn as sns
import warnings

from data import *


np.random.seed(256)
sns.set()
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    print("Started..")
    fs = 200
    lowcut = 0.05 * 3.3  # 9.9 beats per minute
    highcut = 15  # 900 beats per minute
    data = Data("data", "Lviv-Biometric-Data-Set")
    data.describe(user_id = 1)
    data_raw = data.load_raw(user_id = 1, sample_id = 1)
    print("Done!!")



