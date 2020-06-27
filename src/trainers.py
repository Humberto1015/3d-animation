import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.append('./src/')
sys.path.append('./tools/')
import utils
import itertools
from datasets import ACAPData
from models import Encoder, Decoder, Discriminator, Encoder2, Decoder2, Discriminator2
from torch.autograd import Variable

class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__()
        self.opt = opt
        #self.start_visdom()
        self.reset_epoch()

    def start_visdom(self):
        self.vis = utils.Visualizer(port = 8888)

    def increment_epoch(self):
        self.epoch = self.epoch + 1

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = 0

class AutoEncoderTrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.start_visdom()

    def start_visdom(self):
        self.vis = utils.Visualizer(env = 'AutoEncoder Training', port = 8888)

    def build_network(self):
        print ('[info] Build the network architecture')
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.cuda()
        self.decoder.cuda()

    def build_optimizer(self):
        print ('[info] Build the optimizer')
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr = self.opt.learning_rate)

    def build_dataset_train(self):
        train_data = ACAPData(mode = 'train', name = 'SMPL')
        self.num_train_data = len(train_data)
        print ('[info] Number of training samples = ', self.num_train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.opt.batch_size, shuffle = True)

    def build_dataset_valid(self):
        valid_data = ACAPData(mode = 'valid', name = 'SMPL')
        self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 128, shuffle = True)

    def build_losses(self):
        print ('[info] Build the loss functions')
        self.mseLoss = torch.nn.MSELoss()

    def print_iteration_stats(self):
        """
        print stats at each iteration
        """
        print ('\r[Epoch %d] [Iteration %d/%d] Loss = %f' % (
            self.epoch,
            self.iteration,
            int(self.num_train_data/self.opt.batch_size),
            self.loss_train_total.item()), end = '')

    def train_iteration(self):

        self.encoder.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        input_feat = self.data.cuda()

        z = self.encoder(input_feat)
        recon = self.decoder(z)

        # loss
        self.loss_train_total = self.mseLoss(recon, input_feat)
        self.loss.append(self.loss_train_total.item())

        self.loss_train_total.backward()
        self.optimizer.step()
        self.print_iteration_stats()
        self.increment_iteration()

    def valid_iteration(self):

        self.encoder.eval()
        self.decoder.eval()

        input_feat = self.data.cuda()

        z = self.encoder(input_feat)
        recon = self.decoder(z)

        # loss
        self.loss_train_total = self.mseLoss(recon, input_feat)
        self.loss.append(self.loss_train_total.item())

        self.loss_train_total.backward()
        self.increment_iteration()

    def train_epoch(self):

        self.reset_iteration()
        self.loss = []
        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()
        self.loss = torch.Tensor(self.loss)
        self.loss = torch.mean(self.loss)
        self.vis.draw_line(win = 'Train Loss', x = self.epoch, y = self.loss)

    def valid_epoch(self):
        self.reset_iteration()
        self.loss = []
        for step, data in enumerate(self.valid_loader):
            self.data = data
            self.valid_iteration()

        self.loss = torch.Tensor(self.loss)
        self.loss = torch.mean(self.loss)
        self.vis.draw_line(win = 'Valid Loss', x = self.epoch, y = self.loss)

    def save_network(self):
        print("\n[info] saving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")

class AAETrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.start_visdom()

    def start_visdom(self):
        self.vis = utils.Visualizer(env = 'Adversarial AutoEncoder Training', port = 8888)

    def build_network(self):
        print ('[info] Build the network architecture')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        self.encoder.cuda()
        self.decoder.cuda()
        self.discriminator.cuda()

    def build_optimizer(self):
        print ('[info] Build the optimizer')
        self.optim_dis = optim.SGD(self.discriminator.parameters(), lr = self.opt.learning_rate)
        self.optim_AE = optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr = self.opt.learning_rate)

    def build_dataset_train(self):
        train_data = ACAPData(mode = 'train', name = 'SMPL')
        self.num_train_data = len(train_data)
        print ('[info] Number of training samples = ', self.num_train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.opt.batch_size, shuffle = True)

    def build_dataset_valid(self):
        valid_data = ACAPData(mode = 'valid', name = 'SMPL')
        self.num_valid_data = len(valid_data)
        print ('[info] Number of training samples = ', self.num_valid_data)
        self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 128, shuffle = True)

    def build_losses(self):
        print ('[info] Build the loss functions')
        self.mseLoss = torch.nn.MSELoss()
        self.ganLoss = torch.nn.BCELoss()

    def print_iteration_stats(self):
        """
        print stats at each iteration
        """
        print ('\r[Epoch %d] [Iteration %d/%d] enc = %f dis = %f rec = %f' % (
            self.epoch,
            self.iteration,
            int(self.num_train_data/self.opt.batch_size),
            self.enc_loss.item(),
            self.dis_loss.item(),
            self.rec_loss.item()), end = '')

    def sample_from_sphere(self, batch_size):

        # step 1. Sample vectors from arbitrary prior, N(0, I) here
        rand_pts = torch.randn(batch_size, 128)

        # step 2. Centerization
        rand_pts -= rand_pts.mean(0)

        # step 3. Spherization
        for i in range(batch_size):
            rand_pts[i] /= rand_pts[i].norm()
        
        return rand_pts

    def train_iteration(self):

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        x = self.data.cuda()
        z = self.encoder(x)

        ''' Discriminator '''
        # sample from N(0, I)
        #z_real = Variable(torch.randn(z.size(0), z.size(1))).cuda()

        # sample surface points from a sphere
        z_real = self.sample_from_sphere(z.size(0)).cuda()

        y_real = Variable(torch.ones(z.size(0))).cuda()
        dis_real_loss = self.ganLoss(self.discriminator(z_real).view(-1), y_real)

        y_fake = Variable(torch.zeros(z.size(0))).cuda()
        dis_fake_loss = self.ganLoss(self.discriminator(z).view(-1), y_fake)

        self.optim_dis.zero_grad()
        self.dis_loss = 0.5 * (dis_fake_loss + dis_real_loss)
        self.dis_loss.backward(retain_graph = True)
        self.optim_dis.step()
        self.dis_losses.append(self.dis_loss.item())

        ''' Autoencoder '''
        # Encoder hopes to generate latent vectors that are closed to prior.
        y_real = Variable(torch.ones(z.size(0))).cuda()
        self.enc_loss= self.ganLoss(self.discriminator(z).view(-1), y_real)

        # Decoder hopes to make the reconstruction as similar to input as possible.
        rec = self.decoder(z)
        self.rec_loss = self.mseLoss(rec, x)

        self.EG_loss = self.enc_loss + self.rec_loss

        self.optim_AE.zero_grad()
        self.EG_loss.backward()
        self.optim_AE.step()

        self.enc_losses.append(self.enc_loss.item())
        self.rec_losses.append(self.rec_loss.item())

        self.print_iteration_stats()
        self.increment_iteration()

    def train_epoch(self):

        self.reset_iteration()
        self.dis_losses = []
        self.enc_losses = []
        self.rec_losses = []
        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()

        self.dis_losses = torch.Tensor(self.dis_losses)
        self.dis_losses = torch.mean(self.dis_losses)

        self.enc_losses = torch.Tensor(self.enc_losses)
        self.enc_losses = torch.mean(self.enc_losses)

        self.rec_losses = torch.Tensor(self.rec_losses)
        self.rec_losses = torch.mean(self.rec_losses)

        self.vis.draw_line(win = 'Encoder Loss', x = self.epoch, y = self.enc_losses)
        self.vis.draw_line(win = 'Discriminator Loss', x = self.epoch, y = self.dis_losses)
        self.vis.draw_line(win = 'Reconstruction Loss', x = self.epoch, y = self.rec_losses)
    
    def valid_iteration(self):

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        x = self.data.cuda()
        z = self.encoder(x)
        recon = self.decoder(z)

        # loss
        rec_loss = self.mseLoss(recon, x)
        self.rec_loss.append(rec_loss.item())
        self.increment_iteration()

    def valid_epoch(self):
        self.reset_iteration()
        self.rec_loss = []
        for step, data in enumerate(self.valid_loader):
            self.data = data
            self.valid_iteration()

        self.rec_loss = torch.Tensor(self.rec_loss)
        self.rec_loss = torch.mean(self.rec_loss)
        self.vis.draw_line(win = 'Valid reconstruction loss', x = self.epoch, y = self.rec_loss)
    
    def save_network(self):
        print("\n[info] saving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")
        torch.save(self.discriminator.state_dict(), f"{self.opt.save_path}/Discriminator.pth")
