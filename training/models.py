import torch
import torch.nn as nn
import torch.nn.functional as F

""" Convolutional Adversarial Autoencoder """

class Encoder(nn.Module):
    def __init__(self, z_dim = 128):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(9, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(512, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.LeakyReLU(0.2, inplace = True),
        )
    
    # encoding function for a batch of training process
    def forward(self, x):
        x = x.view(x.size(0), -1, 9)
        x = x.permute(0, 2, 1).contiguous()
        x = self.mlp(x)
        x = x.max(dim=2)[0]
        x = self.linear(x)

        return x

class Decoder(nn.Module):
    def __init__(self, num_verts = 6890, z_dim = 128):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(1024, num_verts * 9),
            nn.BatchNorm1d(num_verts * 9),
            nn.Tanh()
        )

    # decoding function for a batch of training process
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 9, -1)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        return x

class Discriminator(nn.Module):

    def __init__(self, input_dim = 128):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)


if __name__ == '__main__':
    
    encoder = Encoder()
    print (encoder)

    decoder = Decoder()
    print (decoder)

    discriminator = Discriminator()
    print (discriminator)