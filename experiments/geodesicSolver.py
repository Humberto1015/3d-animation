import torch
import torch.autograd
import numpy as np

from torch.autograd.functional import jacobian

class GeodesicSolver():
    def __init__(self, opt):
        self.in_betweens = opt.in_betweens
        self.loss = 1000000
        self.last_loss = 100000
        self.gradients = None
        self.alpha = 0.001

    def load_networks(self, encoder, decoder):

        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()

    def init_path(self, s, t):

        print ('[info] Initializing the geodesic path...')

        ''' start point '''
        s = np.load('./ACAP-data/SMPL/' + str(s) + '_norm.npy').astype(np.float32)
        s = torch.from_numpy(s).cuda()
        z_s = self.encoder.encode(s)
        #z_s = torch.randn(128).cuda()

        ''' end point '''
        t = np.load('./ACAP-data/SMPL/' + str(t) + '_norm.npy').astype(np.float32)
        t = torch.from_numpy(t).cuda()
        z_t = self.encoder.encode(t)
        #z_t = torch.randn(128).cuda()

        ''' initialize a discrete geodesic path by interpolating '''
        Z = []
        delta = (z_t - z_s) / (self.in_betweens + 1)

        Z.append(z_s.unsqueeze(0))

        for i in range(1, self.in_betweens + 1):
            Z.append((z_s + i * delta).unsqueeze(0))
            #Z.append(torch.randn(128).cuda().unsqueeze(0))

        Z.append(z_t.unsqueeze(0))

        Z = torch.cat(Z, 0)
        Z = Z.detach().cpu()

        self.Z = Z

    def get_loss(self):
        delta_t = 1 / (self.in_betweens + 1)

        loss = 0
        gradients = []
        Z = self.Z.cuda()
        for i in range(1, len(Z) - 1):

            #print ('Debug: ', self.decoder.decode(Z[i]).size())

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
        self.last_loss = self.loss
        self.loss = (loss.detach().cpu())
        gradients = torch.cat(gradients, 0)
        self.gradients = gradients.detach().cpu()
        del loss
        torch.cuda.empty_cache()
        del gradients
        torch.cuda.empty_cache()
        del Z
        torch.cuda.empty_cache()

    def update(self):
        for i in range(1, len(self.Z) - 1):
            self.Z[i] = self.Z[i] - self.alpha * self.gradients[i - 1]

    def solve(self):

        # use gradient descent to solve for optimal latent vectors
        print ('[info] Start to optimize!')
        iter = 1
        max_iter = 1000
        while abs(self.loss - self.last_loss) / self.last_loss > 0.01 and iter < max_iter:

            self.get_loss()
            self.update()
            print ('-[Iteration %d] loss = %f' % (iter, self.loss.item()))
            iter = iter + 1

        return self.Z.numpy()
