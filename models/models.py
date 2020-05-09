import torch
import torch.nn as nn

class RIMDAutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):

        super(RIMDAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

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

            nn.Linear(64, 1)
        )

    def forward(self, x):

        return self.model(x)


def test_AE():
    AE = RIMDAutoEncoder(input_dim = 60012, hidden_dim = 300, latent_dim = 128)
    fake_data = torch.rand(32, 60012)

    z, recon = AE(fake_data)

    print (z.size(), recon.size())

def test_discriminator():
    D = Discriminator()
    fake_data = torch.rand(32, 128)

    logit = D(fake_data)

    print (logit.size())


if __name__ == '__main__':
    test_AE()
    test_discriminator()
