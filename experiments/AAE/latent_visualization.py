import sys
sys.path.append('./utils/')
sys.path.append('./src/')

from models import Encoder, Decoder
from feature2rimd import RIMDTransformer
from datasets import AnimalRIMD, SmplRIMD
from visualizer import Visualizer
from sklearn.manifold import TSNE

import torch
import struct
import numpy as np
import argparse

if __name__ == '__main__':

    # setup the network
    data = SmplRIMD()
    feat_dim = data.__getitem__(0).shape[0]
    data_loader = torch.utils.data.DataLoader(data, batch_size = 128, shuffle = True)
    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)
    encoder.load_state_dict(torch.load('./trained_weights/AdversarialAutoEncoder/Encoder.pth'))
    decoder.load_state_dict(torch.load('./trained_weights/AdversarialAutoEncoder/Decoder.pth'))
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()

    z_vectors = []
    for step, data in enumerate(data_loader):
        z = encoder(data.cuda())
        z = z.detach().cpu().numpy()
        for v in z:
            z_vectors.append(v)

        if step == 20:
            break
    z_vectors = np.array(z_vectors)
    print (z_vectors.shape)
    # t-sne
    embedded = TSNE(n_components = 2).fit_transform(z_vectors)

    vis = Visualizer(env = 'Latent Space Visualization', port = 8888)
    vis.draw_2d_points(win = 'Latent', verts = embedded, color = np.array([[255, 0, 0]]))
