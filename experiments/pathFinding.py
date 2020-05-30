import sys
sys.path.append('./tools/')
sys.path.append('./src/')
import utils
import argparse
import torch
import torch.autograd
from torch.autograd.functional import jacobian
import numpy as np
from models import Encoder, Decoder
from feature2rimd import RIMDTransformer
from datasets import SmplRIMD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class GeodesicSolver():

    def __init__(self, opt):
        self.in_betweens = opt.in_betweens
        self.loss = 100
        self.gradients = None

    def load_networks(self, encoder, decoder):

        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()

    def init_path(self, s, t):

        print ('Initializing the geodesic path...')
        ''' start point '''
        s = np.load('./rimd-feature/SMPL/' + str(s) + '_norm.npy').astype(np.float32)
        s = torch.from_numpy(s).cuda()
        z_s = self.encoder.encode(s)

        ''' end point '''
        t = np.load('./rimd-feature/SMPL/' + str(t) + '_norm.npy').astype(np.float32)
        t = torch.from_numpy(t).cuda()
        z_t = self.encoder.encode(t)

        ''' initialize a discrete geodesic path by interpolating '''
        Z = []
        delta = (z_t - z_s) / (self.in_betweens + 1)
        for i in range(self.in_betweens + 2):
            Z.append((z_s + i * delta).unsqueeze(0))
        Z = torch.cat(Z, 0)
        Z = Z.detach().cpu()

        self.Z = Z

    def get_loss(self):

        delta_t = 1 / (self.in_betweens + 1)

        loss = 0
        gradients = []
        Z = self.Z.cuda()
        for i in range(1, len(Z) - 1):
            J_h = jacobian(self.encoder.encode, self.decoder.decode(Z[i]))

            term_A = self.decoder.decode(Z[i + 1])
            term_B = self.decoder.decode(Z[i])
            term_C = self.decoder.decode(Z[i - 1])
            term_total = (term_A - 2 * term_B + term_C).unsqueeze(1)

            grad = (-1 / delta_t) * torch.mm(J_h, term_total)
            grad = grad.squeeze(1)
            loss = loss + torch.dot(grad, grad)
            gradients.append(grad.unsqueeze(0))


        ''' [Important] Release GPU memory '''
        self.Z = Z.detach().cpu()
        self.loss = (loss.detach().cpu()) / self.in_betweens
        gradients = torch.cat(gradients, 0)
        self.gradients = gradients.detach().cpu()
        del loss
        torch.cuda.empty_cache()
        del gradients
        torch.cuda.empty_cache()
        del Z
        torch.cuda.empty_cache()

        print ('Loss = ', self.loss)

    def update(self):

        alpha = 0.01
        for i in range(1, len(self.Z) - 1):
            self.Z[i] = self.Z[i] - alpha * self.gradients[i - 1]

    def solve(self):
        # use gradient descent to solve for optimal latent vectors
        print ('Start to optimize!')
        iter = 1
        while self.loss > 0.1:
            print ('[Iteration %d]' % iter)
            self.get_loss()
            self.update()
            iter = iter + 1
        print ('Done.')

        return self.Z.numpy()

def linear_interpolation(opt, encoder, decoder):

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    s = np.load('./rimd-feature/SMPL/' + str(opt.s) + '_norm.npy').astype(np.float32)
    s = torch.from_numpy(s).cuda()
    z_s = encoder.encode(s)

    ''' end point '''
    t = np.load('./rimd-feature/SMPL/' + str(opt.t) + '_norm.npy').astype(np.float32)
    t = torch.from_numpy(t).cuda()
    z_t = encoder.encode(t)

    ''' linear interpolation '''
    Z = []
    delta = (z_t - z_s) / (opt.in_betweens + 1)
    for i in range(opt.in_betweens + 2):
        Z.append((z_s + i * delta).unsqueeze(0))
    Z = torch.cat(Z, 0)
    Z = Z.detach().cpu()

    return Z.numpy()

def geodesic_interpolation(opt, encoder, decoder):

    solver = GeodesicSolver(opt)
    solver.load_networks(encoder, decoder)
    solver.init_path(opt.s, opt.t)

    return solver.solve()

def visualize_path(embedded, seq_linear, seq_geodesic):
    pca = PCA(n_components = 2)
    pca.fit(embedded)

    embedded_2d = pca.transform(embedded)
    xs, ys = embedded_2d.T
    plt.scatter(xs, ys, c = 'r')

    seq_linear_2d = pca.transform(seq_linear)
    xs, ys = seq_linear_2d.T
    plt.scatter(xs, ys, c = 'b')

    seq_geodesic_2d = pca.transform(seq_geodesic)
    xs, ys = seq_geodesic_2d.T
    plt.scatter(xs, ys, c = 'g')

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_weights', type = str, default = './trained_weights/AutoEncoder/Encoder.pth')
    parser.add_argument('--decoder_weights', type = str, default = './trained_weights/AutoEncoder/Decoder.pth')
    parser.add_argument('--s', type = str, help = 'The index of the first model')
    parser.add_argument('--t', type = str, help = 'The index of the second model')
    parser.add_argument('--in_betweens', type = int, help = 'number of frames between the source and target model', default = 9)
    parser.add_argument('--header_path', type = str, default = './rimd-data/SMPL/header.b')
    parser.add_argument('--minima_path', type = str, default = './rimd-feature/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './rimd-feature/SMPL/maxima.npy')
    parser.add_argument('--target_path', type = str, default = './rimd-sequence/')
    opt = parser.parse_args()

    ''' Load networks '''
    print ('Loading networks...')
    data = SmplRIMD()
    data_loader = torch.utils.data.DataLoader(data, batch_size = 128, shuffle = False)
    feat_dim = data.__getitem__(0).shape[0]
    encoder = Encoder(input_dim = feat_dim, hidden_dim = 512, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = feat_dim)
    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.eval()
    decoder.eval()

    seq_linear = linear_interpolation(opt, encoder, decoder)
    seq_geodesic = geodesic_interpolation(opt, encoder, decoder)


    embedded = []
    encoder = encoder.cpu()
    decoder = decoder.cpu()
    for step, data in enumerate(data_loader):
        z = encoder(data)
        z = z.detach().numpy()
        for v in z:
            embedded.append(v)

    visualize_path(embedded, seq_linear, seq_geodesic)

    '''
    # output binary file
    transformer = RIMDTransformer(opt.header_path, opt.minima_path, opt.maxima_path)

    recons = geoSolver.decoder(latent_vectors.cuda())
    for i in range(recons.size(0)):
        rimd = transformer.turn2RIMD((recons[i].detach().cpu().numpy()))
        file_name = opt.target_path + str(i) + '.b'
        utils.write2file(file_name, rimd)
        print ('Saved to %s' % file_name)
    '''
