import sys
sys.path.append('./utils/')
sys.path.append('./src/')

from models import Encoder, Decoder
from feature2rimd import RIMDTransformer
from datasets import AnimalRIMD, SmplRIMD

import torch
import struct
import numpy as np
import argparse


def write2file(file_name, rimd_data):
    fout = open(file_name, 'wb')

    for i in range(len(rimd_data)):

        one_ring = rimd_data[i]

        # dRij
        for j in range(len(one_ring) - 1):
            for k in range(3):
                for l in range(3):
                    fout.write(struct.pack('<f', one_ring[j][k][l]))
        # Si
        for j in range(3):
            for k in range(3):
                fout.write(struct.pack('<f', one_ring[-1][j][k]))
    fout.close()

if __name__ == '__main__':
    # setup the network
    data = SmplRIMD()
    feat_dim = data.__getitem__(0).shape[0]
    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)
    encoder.load_state_dict(torch.load('./trained_weights/AdversarialAutoEncoder/Encoder.pth'))
    decoder.load_state_dict(torch.load('./trained_weights/AdversarialAutoEncoder/Decoder.pth'))
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()

    # setup the transformer (feature -> rimd data)
    header_path = './rimd-data/SMPL/header.b'
    minima_path = './rimd-feature/SMPL/minima.npy'
    maxima_path = './rimd-feature/SMPL/maxima.npy'
    transformer = RIMDTransformer(header_path, minima_path, maxima_path)


    parser = argparse.ArgumentParser()
    parser.add_argument('--m0', type = str, help = 'The index of the first model')
    parser.add_argument('--m1', type = str, help = 'The index of the second model')
    opt = parser.parse_args()

    # start to interpolate
    feat_0 = np.load('./rimd-feature/SMPL/' + str(opt.m0) + '_norm.npy')
    feat_1 = np.load('./rimd-feature/SMPL/' + str(opt.m1) + '_norm.npy')

    feat_0 = (torch.from_numpy(feat_0.astype(np.float32)).cuda()).unsqueeze(0)
    feat_1 = (torch.from_numpy(feat_1.astype(np.float32)).cuda()).unsqueeze(0)


    feat_0.expand(64, feat_0.size(1))
    feat_1.expand(64, feat_1.size(1))

    z_0 = encoder(feat_0)
    z_1 = encoder(feat_1)


    target_dir = './interpolations/'
    num_in_betweens = 8
    step = (z_1 - z_0) / (num_in_betweens + 1)
    for i in range(num_in_betweens + 1):
        z_mid = z_0 + step * i
        f_recon = decoder(z_mid)
        rimd_recon = transformer.turn2RIMD((f_recon[0].detach().cpu().numpy()))
        write2file(target_dir + str(i) + '.b', rimd_recon)
