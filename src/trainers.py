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
from datasets import SmplRIMD
from models import Encoder, Decoder, Discriminator
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
        print ('- Build the network architecture')
        self.encoder = Encoder(input_dim = self.feat_dim, hidden_dim = 512, latent_dim = 128)
        self.decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = self.feat_dim)

        self.encoder.cuda()
        self.decoder.cuda()

    def build_optimizer(self):
        print ('- Build the optimizer')
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr = self.opt.learning_rate)

    def build_dataset_train(self):
        train_data = SmplRIMD()
        self.feat_dim = train_data.__getitem__(0).shape[0]
        print ('Input feature size = ', self.feat_dim)
        self.num_train_data = len(train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.opt.batch_size, shuffle = True, num_workers = self.opt.workers)

    def build_losses(self):
        print ('- Build the loss functions')
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
        """ implementation of iteration computation """
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

    def train_epoch(self):

        self.reset_iteration()
        self.loss = []
        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()
        self.loss = torch.Tensor(self.loss)
        self.loss = torch.mean(self.loss)
        self.vis.draw_line(win = 'Loss', x = self.epoch, y = self.loss)

    def save_network(self):
        print("\nsaving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")

class AdversarialAutoEncoderTrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.start_visdom()

    def start_visdom(self):
        self.vis = utils.Visualizer(env = 'Adversarial AutoEncoder Training', port = 8888)

    def build_network(self):
        print ('- Build the network architecture')
        self.encoder = Encoder(input_dim = self.feat_dim, hidden_dim = 512, latent_dim = 128)
        self.decoder = Decoder(latent_dim = 128, hidden_dim = 512, output_dim = self.feat_dim)
        self.discriminator = Discriminator()

        self.encoder.cuda()
        self.decoder.cuda()
        self.discriminator.cuda()

    def build_optimizer(self):
        print ('- Build the optimizer')
        self.optim_dis = optim.Adam(self.discriminator.parameters(), lr = self.opt.learning_rate, betas = (self.opt.beta1, self.opt.beta2))
        self.optim_dec = optim.Adam(self.decoder.parameters(), lr = self.opt.learning_rate, betas = (self.opt.beta1, self.opt.beta2))
        self.optim_enc = optim.Adam(self.encoder.parameters(), lr = self.opt.learning_rate, betas = (self.opt.beta1, self.opt.beta2))

    def build_dataset_train(self):
        train_data = SmplRIMD()
        self.feat_dim = train_data.__getitem__(0).shape[0]
        self.num_train_data = len(train_data)
        print ('Number of training samples = ', self.num_train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.opt.batch_size, shuffle = True, num_workers = self.opt.workers)

    def build_losses(self):
        print ('- Build the loss functions')
        self.ganLoss = torch.nn.BCEWithLogitsLoss()
        self.recLoss = torch.nn.MSELoss()

    def print_iteration_stats(self):
        """
        print stats at each iteration
        """
        print ('\r[Epoch %d] [Iteration %d/%d] Loss_Enc = %f Loss_Dec = %f Loss_Dis = %f' % (
            self.epoch,
            self.iteration,
            int(self.num_train_data/self.opt.batch_size),
            self.loss_enc.item(),
            self.loss_dec.item(),
            self.loss_dis.item()), end = '')

    def train_iteration(self):
        """ implementation of iteration computation """
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        batch_size = self.data.size(0)

        x = self.data.cuda()
        z = self.encoder(x)

        ''' Discriminator '''
        z_real = torch.randn(batch_size, z.size(1)).cuda()
        y_real = torch.ones(batch_size).cuda()
        dis_real_loss = self.ganLoss(self.discriminator(z_real).view(-1), y_real)

        y_fake = torch.zeros(batch_size).cuda()
        dis_fake_loss = self.ganLoss(self.discriminator(z).view(-1), y_fake)

        self.optim_dis.zero_grad()
        self.loss_dis = dis_fake_loss + dis_real_loss
        self.loss_dis.backward(retain_graph = True)
        self.optim_dis.step()
        self.dis_losses.append(self.loss_dis.item())
        ''' ------------- '''

        '''    Encoder    '''
        y_real = torch.ones(batch_size).cuda()
        enc_gan_loss = self.ganLoss(self.discriminator(z).view(-1), y_real)

        self.loss_enc = enc_gan_loss
        self.optim_enc.zero_grad()
        self.loss_enc.backward(retain_graph = True)
        self.optim_enc.step()
        self.enc_losses.append(self.loss_enc.item())
        ''' ------------- '''

        '''    Decoder    '''
        rec = self.decoder(z)
        self.loss_dec = self.recLoss(rec, x)
        self.optim_dec.zero_grad()
        self.loss_dec.backward()
        self.optim_dec.step()
        self.dec_losses.append(self.loss_dec.item())
        ''' ------------- '''

        self.print_iteration_stats()
        self.increment_iteration()

    def train_epoch(self):
        self.reset_iteration()

        self.dis_losses = []
        self.enc_losses = []
        self.dec_losses = []

        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()

        loss_enc = torch.Tensor(self.enc_losses)
        loss_enc = torch.mean(loss_enc)

        loss_dec = torch.Tensor(self.dec_losses)
        loss_dec = torch.mean(loss_dec)

        loss_dis = torch.Tensor(self.dis_losses)
        loss_dis = torch.mean(loss_dis)

        self.vis.draw_line(win = 'Encoder Loss', x = self.epoch, y = loss_enc)
        self.vis.draw_line(win = 'Decoder Loss', x = self.epoch, y = loss_dec)
        self.vis.draw_line(win = 'Discriminator Loss', x = self.epoch, y = loss_dis)

    def save_network(self):
        print("\nsaving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")
        torch.save(self.discriminator.state_dict(), f"{self.opt.save_path}/Discriminator.pth")
