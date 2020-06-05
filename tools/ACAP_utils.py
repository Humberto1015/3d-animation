import numpy as np

def gen_feature():
    root = './ACAP-data/SMPL/'

    num_samples = 10000

    ''' step 1. Get minima and maxima '''
    minima = None
    maxima = None
    for i in range(num_samples):

        feat = np.load(root + str(i) + '.npy')
        feat = feat.flatten()

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

        feat = np.reshape(feat, (-1, 9))

        file_name = root + str(i) + '_norm'
        np.save(file_name, feat)
        print ('Saved to %s.npy' % file_name)


def back_mapping(feat, minima, maxima):
    a = 0.95
    for i in range(feat.shape[0]):
        min_val = minima[i].copy()
        max_val = maxima[i].copy()
        feat[i] = ((feat[i] + a) / (2.0 * a)) * (max_val - min_val) + min_val

    return feat

def test_recover():

    minima = np.load('./ACAP-data/SMPL/minima.npy')
    maxima = np.load('./ACAP-data/SMPL/maxima.npy')

    for i in range(10):
        feat = np.load('./ACAP-data/SMPL/' + str(i) + '_norm.npy')
        feat_recov = back_mapping(feat.flatten(), minima, maxima)

        np.save('./ACAP-sequence/' + str(i), feat_recov)

if __name__ == '__main__':
    #gen_feature()
    test_recover()
