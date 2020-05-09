import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('./src/')
from datasets import AnimalRIMD
from models import Encoder, Decoder, Discriminator
from torch.autograd import Variable


# basic trainer
class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__()
        self.start_time = time.time()
        self.opt = opt
        self.start_visdom()
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

    def build_network(self):
        print ('- Build the network architecture')
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.cuda()
        self.decoder.cuda()

    def build_optimizer(self):
        print ('- Build the optimizer')
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr = self.opt.learning_rate)

    def build_dataset_train(self):
        train_data = AnimalRIMD(train = True)
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

        self.loss_train_total.backward()
        self.optimizer.step()
        self.print_iteration_stats()


        self.increment_iteration()

    def train_epoch(self):
        self.reset_iteration()
        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()
        self.vis.draw_line(win = 'Loss', epoch = self.epoch, value = self.loss_train_total.item())

    def save_network(self):
        print("\nsaving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")

class AdversarialAutoEncoderTrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

    def build_network(self):
        print ('- Build the network architecture')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        self.encoder.cuda()
        self.decoder.cuda()
        self.discriminator.cuda()

    def build_optimizer(self):
        print ('- Build the optimizer')
        # TODO

    def build_dataset_train(self):
        train_data = AnimalRIMD(train = True)
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
        self.discriminator.train()


        self.optimizer.zero_grad()

        input_feat = self.data.cuda()
        z = self.encoder(input_feat)
        recon = self.decoder(z)

        # loss
        self.loss_train_total = self.mseLoss(recon, input_feat)

        self.loss_train_total.backward()
        self.optimizer.step()
        self.print_iteration_stats()


        self.increment_iteration()

    def train_epoch(self):
        self.reset_iteration()
        for step, data in enumerate(self.train_loader):
            self.data = data
            self.train_iteration()
        self.vis.draw_line(win = 'Loss of source domain training', epoch = self.epoch, value = self.loss_train_total.item())

    def save_network(self):
        print("\nsaving net...")
        torch.save(self.encoder.state_dict(), f"{self.opt.save_path}/Encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.opt.save_path}/Decoder.pth")
        torch.save(self.discriminator.state_dict(), f"{self.opt.save_path}/Discriminator.pth")
