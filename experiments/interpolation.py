import torch
import struct
import numpy as np
import argparse
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
    parser.add_argument('--m0', type = str, help = 'The index of the first model')
    parser.add_argument('--m1', type = str, help = 'The index of the second model')
    parser.add_argument('--num_in_betweens', type = int, help = 'number of frames between the source and target model',default = 3)
    parser.add_argument('--header_path', type = str, default = './rimd-data/SMPL/header.b')
    parser.add_argument('--minima_path', type = str, default = './rimd-feature/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './rimd-feature/SMPL/maxima.npy')
    parser.add_argument('--target_path', type = str, default = './rimd-sequence/')
    opt = parser.parse_args()

    # setup the network
    data = SmplRIMD()
    feat_dim = data.__getitem__(0).shape[0]
    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)

    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.cuda().eval()
    decoder.cuda().eval()

    # setup the transformer (feature -> rimd data)
    transformer = RIMDTransformer(opt.header_path, opt.minima_path, opt.maxima_path)

    # start to interpolate
    feat_0 = np.load('./rimd-feature/SMPL/' + str(opt.m0) + '_norm.npy')
    feat_1 = np.load('./rimd-feature/SMPL/' + str(opt.m1) + '_norm.npy')

    feat_0 = (torch.from_numpy(feat_0.astype(np.float32)).cuda()).unsqueeze(0)
    feat_1 = (torch.from_numpy(feat_1.astype(np.float32)).cuda()).unsqueeze(0)


    feat_0.expand(64, feat_0.size(1))
    feat_1.expand(64, feat_1.size(1))

    z_0 = encoder(feat_0)
    z_1 = encoder(feat_1)


    step = (z_1 - z_0) / (opt.num_in_betweens + 1)
    for i in range(opt.num_in_betweens + 2):
        z_mid = z_0 + step * i
        f_recon = decoder(z_mid)
        rimd_recon = transformer.turn2RIMD((f_recon[0].detach().cpu().numpy()))
        file_name = opt.target_path + str(i) + '.b'
        utils.write2file(file_name, rimd_recon)
        print ('Saved to %s' % file_name)
