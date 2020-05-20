import argparse
import numpy as np
import mlrose
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import sys
sys.path.append('./tools/')
sys.path.append('./src/')
import utils
from models import Encoder, Decoder
from feature2rimd import RIMDTransformer
from datasets import SmplRIMD

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_weights', type = str, default = './trained_weights/AdversarialAutoEncoder/Encoder.pth')
    parser.add_argument('--decoder_weights', type = str, default = './trained_weights/AdversarialAutoEncoder/Decoder.pth')
    parser.add_argument('--header_path', type = str, default = './rimd-data/SMPL/header.b')
    parser.add_argument('--minima_path', type = str, default = './rimd-feature/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './rimd-feature/SMPL/maxima.npy')
    opt = parser.parse_args()

    """ network """
    data = SmplRIMD()
    feat_dim = data.__getitem__(0).shape[0]
    data_loader = torch.utils.data.DataLoader(data, batch_size = 8, shuffle = False, num_workers = 8)
    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)
    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.cuda().eval()
    decoder.cuda().eval()
    """ ======================================== """

    """ Step 1. Specify key frames by user  """

    # file index of the key frame shape
    f_0 = 14
    f_1 = 92
    f_2 = 33

    z_0 = np.load('./rimd-feature/SMPL/' + str(f_0) + '_norm.npy')
    z_1 = np.load('./rimd-feature/SMPL/' + str(f_1) + '_norm.npy')
    z_2 = np.load('./rimd-feature/SMPL/' + str(f_2) + '_norm.npy')

    samples = np.array([z_0, z_1, z_2]).astype(np.float32)
    samples = torch.from_numpy(samples)
    samples = encoder(samples.cuda())
    samples = samples.detach().cpu().numpy()
    """ ======================================== """


    """ == Step 2. Solve the TSP problem ======= """
    num_nodes = samples.shape[0]
    dist = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = np.linalg.norm(samples[i] - samples[j])
            dist.append((i, j, d))

    fitness_dists = mlrose.TravellingSales(distances = dist)
    problem_fit = mlrose.TSPOpt(length = num_nodes, fitness_fn = fitness_dists, maximize = False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)

    print ('The length of Hamilton Curcuit: %f' % best_fitness)

    best_state = best_state.tolist()
    best_state.append(best_state[0])
    print ('Sequence indices: ', best_state)
    """ =========================================="""

    """ == Step 3. Interpolations =============== """
    seq = []
    for i in range(len(best_state)):

        s = samples[best_state[i]]

        if i == len(best_state) - 1:
            t = samples[best_state[0]]
        else:
            t = samples[best_state[i + 1]]

        seq.append(s)

        num_in_betweens = 8
        diff = (t - s) / (num_in_betweens + 1)
        for j in range(1, num_in_betweens + 1):
            mid = s + j * diff
            seq.append(mid)

        seq.append(t)

    seq = np.array(seq)
    """ =========================================="""


    """ ========== Visualization ================ """
    # dimension reduction
    embedded = TSNE(n_components=2).fit_transform(seq)

    # visualization
    xs, ys = embedded.T

    plt.scatter(xs, ys)

    # draw edges
    for i in range(seq.shape[0]):

        if i == seq.shape[0] - 1:
            x = [xs[i], xs[0]]
            y = [ys[i], ys[0]]
        else:
            x = [xs[i], xs[i + 1]]
            y = [ys[i], ys[i + 1]]
        plt.plot(x, y, c = 'black')

    #plt.show()
    """ =========================================="""

    recon = decoder(torch.from_numpy(seq).cuda())
    recon = recon.detach().cpu().numpy()

    transformer = RIMDTransformer(opt.header_path, opt.minima_path, opt.maxima_path)

    for i in range(recon.shape[0]):
        rimd = transformer.turn2RIMD(recon[i])
        utils.write2file('./rimd-sequence/' + str(i) + '.b', rimd)
