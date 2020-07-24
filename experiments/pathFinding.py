import sys
sys.path.append('./tools/')
sys.path.append('./src/')
import utils
import argparse
import torch
import torch.autograd
import numpy as np
from models import Encoder, Decoder
from datasets import ACAPData
from graph import Graph

def back_mapping(feat, minima, maxima):
    a = 0.95
    for i in range(feat.shape[0]):
        min_val = minima[i].copy()
        max_val = maxima[i].copy()
        feat[i] = ((feat[i] + a) / (2.0 * a)) * (max_val - min_val) + min_val

    return feat

if __name__ == '__main__':

    dataset = 'SMPL'

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_weights', type = str, default = './trained_weights/AAE_' + dataset + '/Encoder.pth')
    parser.add_argument('--decoder_weights', type = str, default = './trained_weights/AAE_' + dataset + '/Decoder.pth')
    parser.add_argument('--feature_path', type = str, default = './ACAP-data/' + dataset + '/')
    parser.add_argument('--minima_path', type = str, default = './ACAP-data/' + dataset + '/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './ACAP-data/' + dataset + '/maxima.npy')
    parser.add_argument('--target_path', type = str, default = './ACAP-sequence/')
    parser.add_argument('--mode', type = str)
    opt = parser.parse_args()

    ''' Load networks '''
    if dataset == 'SMPL':
        num_verts = 6890
    elif dataset == 'all_animals':
        num_verts = 3889

    encoder = Encoder()
    decoder = Decoder(num_verts = num_verts)

    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.eval()
    decoder.eval()
    
    s = 0
    t = 82


    # Implementation of Linear Interpolation
    if opt.mode == 'linear':

        f_s = np.load(opt.feature_path + str(s) + '_norm.npy').astype(np.float32)
        f_s = torch.from_numpy(f_s)
        z_s = encoder(f_s.unsqueeze(0))
        z_s = z_s.squeeze(0).detach().numpy()

        f_t = np.load(opt.feature_path + str(t) + '_norm.npy').astype(np.float32)
        f_t = torch.from_numpy(f_t)
        z_t = encoder(f_t.unsqueeze(0))
        z_t = z_t.squeeze(0).detach().numpy()

        t_vals = np.linspace(0, 1, 61)

        nodes = []
        for t in t_vals:
            nodes.append(z_s + t * (z_t - z_s))
        nodes = np.array(nodes)

        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        recons = decoder(torch.from_numpy(nodes))
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().cpu().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)

    # Implementation of path exploration algorithm (without extra keyframes)
    if opt.mode == 'case1':
        
        nodes = []
        for i in range(10000):
            x = np.load(opt.feature_path + str(i) + '_norm.npy').astype(np.float32)
            x = torch.from_numpy(x)
            z = encoder(x.unsqueeze(0))
            nodes.append(z.squeeze(0).detach().numpy())
        nodes = np.array(nodes).astype(np.float32)
            
        g = Graph(nodes)
        g.solve(s = s, t = t)
        seq = g.solution.astype(np.float32)
        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        decoder = decoder
        encoder = encoder
        recons = decoder(torch.from_numpy(seq))
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)
        g.show()
    
    # Implementation of path exploration algorithm (with interested keyframes)
    if opt.mode == 'case2':
        # construct knn-mst graph
        nodes = []
        for i in range(10000):
            x = np.load(opt.feature_path + str(i) + '_norm.npy').astype(np.float32)
            x = torch.from_numpy(x)
            z = encoder(x.unsqueeze(0))
            nodes.append(z.squeeze(0).detach().numpy())
        nodes = np.array(nodes).astype(np.float32)
            
        g = Graph(nodes)

        g.solve(s = 76, t = 0)
        seq_0 = g.solution.astype(np.float32)

        g.solve(s = 0, t = 109)
        seq_1 = g.solution.astype(np.float32)
        seq = np.append(seq_0, seq_1[1:], 0)
        
        minima = np.load(opt.minima_path)
        maxima = np.load(opt.maxima_path)
        decoder = decoder
        encoder = encoder
        recons = decoder(torch.from_numpy(seq))
        for i in range(recons.size(0)):
            feat = back_mapping((recons[i].detach().numpy()).flatten(), minima, maxima)
            file_name = opt.target_path + str(i) + '.npy'
            np.save(file_name, feat)
            print ('[info] Saved to %s' % file_name)
