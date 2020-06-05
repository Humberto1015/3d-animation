import torch
import torch.nn as nn
import torch.nn.functional as F

""" Convolutional Autoencoder """
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(9, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()

            #nn.Conv1d(128, 1024, 1),
            #nn.BatchNorm1d(1024),
            #nn.ReLU()
        )
    def encode(self, x):
        x = x.unsqueeze(0)
        x = self.model(x.permute(0, 2, 1).contiguous()).max(dim=2)[0]
        x = x.squeeze(0)
        return x

    def forward(self, x):
        return self.model(x.permute(0, 2, 1).contiguous()).max(dim=2)[0]

class Decoder2(nn.Module):
    def __init__(self, num_verts = 6890):
        super(Decoder2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, num_verts * 9),
            nn.BatchNorm1d(num_verts * 9),
            nn.Tanh()
        )

    def decode(self, x):
        x = x.unsqueeze(0)
        x = self.model(x)
        x = x.view(x.size(0), 9, -1)
        x = x.permute(0, 2, 1).contiguous()
        x = x.squeeze(0)
        return x

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 9, -1)
        return x.permute(0, 2, 1).contiguous()

""" Fully Connected Autoencoder """
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()


        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.3),
            #nn.Dropout(0.5),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.3)
        )


    # encode only one sample
    def encode(self, x):


        x = x.unsqueeze(0)
        x = self.model(x)
        x = x.squeeze(0)
        '''
        x = x.permute(0, 2, 1).contiguous()
        x = self.model(x)
        '''

        return x

    def forward(self, x):

        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()


        self.model = self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.3),
            #nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Tanh()
        )
        '''

        num_verts = 6890

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, num_verts * 9),
            nn.BatchNorm1d(num_verts * 9),
            nn.Tanh()
        )
        '''


    # decode only one sample
    def decode(self, z):



        z = z.unsqueeze(0)
        #z = z.permute(0, 2, 1).contiguous()
        z = self.model(z)
        #x = x.view(x.size(0), 9, -1)
        z = z.squeeze(0)

        return z

    def forward(self, x):
        x = self.model(x)
        #x = x.view(x.size(0), 9, -1)
        #x = x.permute(0, 2, 1).contiguous()

        return x

class Discriminator(nn.Module):

    def __init__(self, input_dim = 128):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.3),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.3),

            nn.Linear(512, 128),
            nn.LeakyReLU(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)


def test_AE():
    encoder = Encoder(input_dim = 60012, hidden_dim = 300, latent_dim = 128)
    decoder = Decoder(latent_dim = 128, hidden_dim = 300, output_dim = 60012)

    fake_data = torch.rand(32, 60012)

    z = encoder(fake_data)
    recon = decoder(z)

    print (z.size(), recon.size())

def test_discriminator():
    D = Discriminator()
    fake_data = torch.rand(32, 128)

    logit = D(fake_data)

    print (logit.size())


if __name__ == '__main__':
    test_AE()
    test_discriminator()
