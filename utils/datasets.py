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
        return 5000


class AnimalRIMD(Dataset):
    def __init__(self, train = True):
        self.train = train

        if self.train:
            self.path = './rimd-feature/Animal_all/train/'
        else:
            self.path = './rimd-feature/Animal_all/test/'

    def __getitem__(self, index):


        data = np.load(self.path + str(index) + '_norm.npy')
        data = data.astype(np.float32)
        return data

    def __len__(self):
        if self.train:
            return 25000
        else:
            return 1000

def test_animal():
    dataset = AnimalRIMD(train = False)
    print (dataset.__getitem__(1).shape)

if __name__ == '__main__':
    test_animal()
