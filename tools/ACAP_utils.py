import numpy as np
import os
import argparse

def preprocess(root = './ACAP-data/SMPL'):
    
    num_samples = len(os.listdir(root))

    ''' step 1. Get minima and maxima '''
    minima = None
    maxima = None
    for i in range(num_samples):

        feat = np.load(root + str(i) + '.npy')

        if i == 0:
            minima = feat.copy()
            maxima = feat.copy()
            continue

        for j in range(feat.shape[0]):

            if feat[j] > maxima[j]:
                maxima[j] = feat[j].copy()
            if feat[j] < minima[j]:
                minima[j] = feat[j].copy()

            if maxima[j] == minima[j]:
                print ('The max value equals to the min value!')
                maxima[j] = maxima[j] + 1e-6
                minima[j] = minima[j] - 1e-6

    np.save(root + 'minima', minima)
    np.save(root + 'maxima', maxima)

    ''' step 2. normalize to [-0.95, 0.95] & save as npy '''
    a = 0.95
    for i in range(num_samples):

        feat = np.load(root + str(i) + '.npy')
        num_dim = feat.shape[0]
        for j in range(num_dim):
            min_val = minima[j].copy()
            max_val = maxima[j].copy()

            feat[j] = 2 * a * ((feat[j] - min_val)/ (max_val - min_val)) - a

        file_name = root + str(i) + '_norm'
        np.save(file_name, feat)
        print ('Saved to %s.npy' % file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str, default = './ACAP-data/SMPL')
    opt = parser.parse_args()

    preprocess(opt.path)