from torch.utils.data import Dataset
import numpy as np

class SmplRIMD(Dataset):
    def __init__(self, mode = 'train'):
        #self.path = './rimd-feature/SMPL/'
        self.path = './ACAP-data/SMPL/'
        self.mode = mode

    def __getitem__(self, index):

        if self.mode == 'train':
            offset = 0
        elif self.mode == 'valid':
            offset = 9000

        data = np.load(self.path + str(offset + index) + '_norm.npy')
        data = data.astype(np.float32)
        return data

    def __len__(self):
        if self.mode == 'train':
            return 9000
        elif self.mode == 'valid':
            return 1000
