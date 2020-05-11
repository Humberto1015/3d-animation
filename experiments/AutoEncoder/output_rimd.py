import sys
sys.path.append('./utils/')
sys.path.append('./src/')

from models import Encoder, Decoder
from feature2rimd import RIMDTransformer
from datasets import AnimalRIMD

import torch
import struct
import numpy as np

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
    data = AnimalRIMD(train = False)
    feat_dim = data.__getitem__(0).shape[0]
    data_loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True, num_workers = 8)

    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)

    encoder.load_state_dict(torch.load('./trained_weights/AutoEncoder/Encoder.pth'))
    decoder.load_state_dict(torch.load('./trained_weights/AutoEncoder/Decoder.pth'))

    encoder.cuda()
    decoder.cuda()

    encoder.eval()
    decoder.eval()


    header_path = './rimd-data/Animal_all/test/header.b'
    minima_path = './rimd-feature/Animal_all/test/minima.npy'
    maxima_path = './rimd-feature/Animal_all/test/maxima.npy'


    transformer = RIMDTransformer(header_path, minima_path, maxima_path)


    for step, data in enumerate(data_loader):
        z = encoder(data.cuda())
        recon = decoder(z)


        recon = recon.detach().cpu().numpy()
        for i in range(recon.shape[0]):
            rimd = transformer.turn2RIMD(recon[i])
            write2file(str(i) + '.b', rimd)

        break
