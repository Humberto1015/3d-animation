import sys
sys.path.append('./tools/')
sys.path.append('./src/')
import utils
import argparse
import torch
import torch.autograd
import numpy as np
from models import Encoder, Decoder, Encoder2, Decoder2
from datasets import ACAPData
from geodesicSolver import GeodesicSolver

from graph import Graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shortest_path(encoder, decoder):
    g = Graph(encoder, decoder)
    g.solvePath()
    g.show()

def sample_from_gaussian(opt, encoder, decoder):
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    z = torch.randn(opt.in_betweens + 2, 128)

    return z.detach().numpy()

def linear_interpolation(opt, encoder, decoder):

    print ('[info] Processing linear interpolation...')

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    s = np.load('./ACAP-data/SMPL/' + str(opt.s) + '_norm.npy').astype(np.float32)
    s = torch.from_numpy(s).cuda()
    z_s = encoder(s.unsqueeze(0))
    
    ''' end point '''
    t = np.load('./ACAP-data/SMPL/' + str(opt.t) + '_norm.npy').astype(np.float32)
    t = torch.from_numpy(t).cuda()
    z_t = encoder(t.unsqueeze(0))

    #t_vals = np.linspace(0, 1, opt.in_betweens + 2)
    #print (t_vals)
    Z = []

    #for t in t_vals:
    #    Z.append(slerp(z_s, z_t, t))

    #Z = np.array(Z)

    delta = (z_t - z_s) / (opt.in_betweens + 1)
    for i in range(opt.in_betweens + 2):
        Z.append((z_s + i * delta).unsqueeze(0))
    Z = torch.cat(Z, 0)
    Z = Z.detach().cpu()
    Z = Z.squeeze(1).numpy()

    for z in Z:
        print (np.linalg.norm(z))
    
    return Z

def back_mapping(feat, minima, maxima):
    a = 0.95
    for i in range(feat.shape[0]):
        min_val = minima[i].copy()
        max_val = maxima[i].copy()
        feat[i] = ((feat[i] + a) / (2.0 * a)) * (max_val - min_val) + min_val

    return feat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_weights', type = str, default = './trained_weights/AAE_normal/Encoder.pth')
    parser.add_argument('--decoder_weights', type = str, default = './trained_weights/AAE_normal/Decoder.pth')
    parser.add_argument('--s', type = str, help = 'The index of the first model')
    parser.add_argument('--t', type = str, help = 'The index of the second model')
    parser.add_argument('--in_betweens', type = int, help = 'number of frames between the source and target model', default = 18)
    parser.add_argument('--minima_path', type = str, default = './ACAP-data/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './ACAP-data/SMPL/maxima.npy')
    parser.add_argument('--target_path', type = str, default = './ACAP-sequence/')
    parser.add_argument('--mode', type = str)
    opt = parser.parse_args()

    ''' Load networks '''
    encoder = Encoder()
    decoder = Decoder()

    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.eval()
    decoder.eval()

    if opt.mode == 'gaussian':
        seq_gaussian = sample_from_gaussian(opt, encoder, decoder)
        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        recons = decoder(torch.from_numpy(seq_gaussian).cuda())
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().cpu().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)

    if opt.mode == 'output_linear':
        seq_linear = linear_interpolation(opt, encoder, decoder)
        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        recons = decoder(torch.from_numpy(seq_linear).cuda())
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().cpu().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)

    if opt.mode == 'output_geodesic':
        seq_geodesic = geodesic_interpolation(opt, encoder, decoder)
        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        recons = decoder(torch.from_numpy(seq_geodesic).cuda())
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().cpu().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)

    if opt.mode == 'shortest_path':
        shortest_path(encoder, decoder)
    
    # check if the learned distribution covers a sphere surface
    if opt.mode == 'debug':

        data = ACAPData()
        data_loader = torch.utils.data.DataLoader(data, batch_size = 64, shuffle = False)

        embedded = []
        for step, samples in enumerate(data_loader):
            codes = encoder(samples)
            for code in codes:
                embedded.append(code.detach().numpy())
            
            if step == 50:
                break
            
        embedded = np.array(embedded)

        # vis
        embedded = np.array(embedded)
        xs, ys, zs = embedded.T

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim3d(-1.5, 1.5)
        ax.set_ylim3d(-1.5, 1.5)
        ax.set_zlim3d(-1, 1)

        ax.scatter(xs, ys, zs)
        plt.show()