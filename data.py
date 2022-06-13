import os
import numpy as np


class Data:
    def __init__(self, dir, data_folder):
        self.data_path = os.path.join(dir, data_folder)
        self.data_dirs = os.listdir(self.data_path)
        self.data_dirs.remove("LICENSE")
        self.data_dirs.remove("README.md")
        self.user_id = None
        self.user_path = None
        self.user_info = None
        self.sample_id = None

    def describe(self, user_id):
        self.user_id = user_id
        self.user_path = os.path.join(self.data_path, "user", str(self.user_id))
        self.user_info = os.listdir(self.user_path)
        print("_______________________________________")
        print("Description:\n user{} has {}, ecg samples {}".format(str(self.user_id), len(self.user_info), self.user_info))
        return

    def load_raw(self, sample_id):
        self.sample_id = sample_id
        data = np.load(os.path.join(self.data_path, self.user_info[self.sample_id]))
        return data

    
        
