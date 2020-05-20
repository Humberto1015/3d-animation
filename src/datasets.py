from torch.utils.data import Dataset
import numpy as np

class SmplRIMD(Dataset):
    def __init__(self):
        self.path = './rimd-feature/SMPL/'

    def __getitem__(self, index):
        data = np.load(self.path + str(index) + '_norm.npy')
        data = data.astype(np.float32)
        return data

    def __len__(self):
        return 10000
