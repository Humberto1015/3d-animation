import sys
sys.path.append('./tools/')
sys.path.append('./src/')
import utils
import argparse
import torch
import torch.autograd
from torch.autograd.functional import jacobian
import numpy as np
from models import Encoder2, Decoder2
from datasets import SmplRIMD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GeodesicSolver():

    def __init__(self, opt):
        self.in_betweens = opt.in_betweens
        self.loss = 100
        self.gradients = None

    def load_networks(self, encoder, decoder):

        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()

    def init_path(self, s, t):

        print ('[info] Initializing the geodesic path...')
        ''' start point '''
        s = np.load('./ACAP-data/SMPL/' + str(s) + '_norm.npy').astype(np.float32)
        s = torch.from_numpy(s).cuda()
        z_s = self.encoder.encode(s)

        ''' end point '''
        t = np.load('./ACAP-data/SMPL/' + str(t) + '_norm.npy').astype(np.float32)
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
            print (J_h.size())

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

    def update(self):

        alpha = 0.01
        for i in range(1, len(self.Z) - 1):
            self.Z[i] = self.Z[i] - alpha * self.gradients[i - 1]

    def solve(self):
        # use gradient descent to solve for optimal latent vectors
        print ('[info] Start to optimize!')
        iter = 1
        while self.loss > 0.1:

            self.get_loss()
            self.update()
            print ('-[Iteration %d] loss = %f' % (iter, self.loss.item()))
            iter = iter + 1

        return self.Z.numpy()

def linear_interpolation(opt, encoder, decoder):

    print ('[info] Processing linear interpolation...')

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    s = np.load('./ACAP-data/SMPL/' + str(opt.s) + '_norm.npy').astype(np.float32)
    s = torch.from_numpy(s).cuda()
    z_s = encoder.encode(s)

    ''' end point '''
    t = np.load('./ACAP-data/SMPL/' + str(opt.t) + '_norm.npy').astype(np.float32)
    t = torch.from_numpy(t).cuda()
    z_t = encoder.encode(t)

    ''' linear interpolation '''
    Z = []
    delta = (z_t - z_s) / (opt.in_betweens + 1)
    for i in range(opt.in_betweens + 2):
        Z.append((z_s + i * delta).unsqueeze(0))
    Z = torch.cat(Z, 0)
    Z = Z.detach().cpu()

    print ('[info] Done.')

    return Z.numpy()

def geodesic_interpolation(opt, encoder, decoder):

    print ('[info] Processing geodesic interpolation')

    solver = GeodesicSolver(opt)
    solver.load_networks(encoder, decoder)
    solver.init_path(opt.s, opt.t)

    return solver.solve()

def visualize_path(embedded, seq_linear, seq_geodesic):
    pca = PCA(n_components = 2)
    pca.fit(embedded)

    # visualize
    #fig = plt.figure()
    #ax = Axes3D(fig)

    embedded_2d = pca.transform(embedded)
    xs, ys= embedded_2d.T
    plt.scatter(xs, ys, c = 'r')

    seq_linear_2d = pca.transform(seq_linear)
    xs, ys = seq_linear_2d.T
    plt.scatter(xs, ys, c = 'b')

    # draw edges
    for i in range(xs.shape[0] - 1):
        plt.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], c = 'b')

    seq_geodesic_2d = pca.transform(seq_geodesic)
    xs, ys = seq_geodesic_2d.T
    plt.scatter(xs, ys, c = 'g')
    # draw edges
    for i in range(xs.shape[0] - 1):
        plt.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], c = 'g')

    plt.show()

def debug():
    pass

def back_mapping(feat, minima, maxima):
    a = 0.95
    for i in range(feat.shape[0]):
        min_val = minima[i].copy()
        max_val = maxima[i].copy()
        feat[i] = ((feat[i] + a) / (2.0 * a)) * (max_val - min_val) + min_val

    return feat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_weights', type = str, default = './trained_weights/AutoEncoder/Encoder.pth')
    parser.add_argument('--decoder_weights', type = str, default = './trained_weights/AutoEncoder/Decoder.pth')
    parser.add_argument('--s', type = str, help = 'The index of the first model')
    parser.add_argument('--t', type = str, help = 'The index of the second model')
    parser.add_argument('--in_betweens', type = int, help = 'number of frames between the source and target model', default = 9)
    parser.add_argument('--minima_path', type = str, default = './ACAP-data/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './ACAP-data/SMPL/maxima.npy')
    parser.add_argument('--target_path', type = str, default = './ACAP-sequence/')
    parser.add_argument('--mode', type = str)
    opt = parser.parse_args()

    ''' Load networks '''
    data = SmplRIMD()
    data_loader = torch.utils.data.DataLoader(data, batch_size = 128, shuffle = False)
    feat_dim = data.__getitem__(0).shape[0]
    encoder = Encoder2()
    decoder = Decoder2()

    encoder.load_state_dict(torch.load(opt.encoder_weights))
    decoder.load_state_dict(torch.load(opt.decoder_weights))
    encoder.eval()
    decoder.eval()

    if opt.mode == 'visualize':
        seq_linear = linear_interpolation(opt, encoder, decoder)
        #seq_geodesic = geodesic_interpolation(opt, encoder, decoder)


        embedded = []
        encoder = encoder.cpu()
        decoder = decoder.cpu()
        for step, data in enumerate(data_loader):
            z = encoder(data)
            z = z.detach().numpy()
            for v in z:
                embedded.append(v)

            if (step == 20):
                break

        visualize_path(embedded, seq_linear, seq_linear)


    if opt.mode == 'debug':
        x = torch.rand(6890, 9)
        x = encoder.encode(x)
        x = decoder.decode(x)
        print (x.size())

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
        converter = utils.Converter(opt)
        recons = decoder(torch.from_numpy(seq_geodesic).cuda())
        for i in range(recons.size(0)):
            rimd = converter.feat2RIMD((recons[i].detach().cpu().numpy()))
            file_name = opt.target_path + str(i) + '.b'
            converter.rimd2file(file_name, rimd)
            print ('[info] Saved to %s' % file_name)
