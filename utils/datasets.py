from torch.utils.data import Dataset
import numpy as np

class AnimalRIMD(Dataset):
    def __init__(self, train = True):
        self.train = train

        if self.train:
            self.path = './rimd-feature/Animal_all/train/'
        else:
            self.path = './rimd-feature/Animal_all/test/'

        self.minima = np.load(self.path + 'minima.npy')
        self.maxima = np.load(self.path + 'maxima.npy')
        self.a = 0.9

    def __getitem__(self, index):


        data = np.load(self.path + str(index) + '_norm.npy')

        # normalize to -0.9 ~ 0.9
        #for i in range(data.shape[0]):
        #    min_value = self.minima[i].copy()
        #    max_value = self.maxima[i].copy()
        #    data[i] = 2 * self.a * ((data[i] - min_value) / (max_value - min_value)) - self.a

        data = data.astype(np.float32)
        return data

    def __len__(self):
        if self.train:
            return 25000
        else:
            return 100

def test_animal():
    dataset = AnimalRIMD(train = False)
    print (dataset.__getitem__(1).shape)

if __name__ == '__main__':
    test_animal()
