import torch.nn as nn
import torch.nn.functional as f
import torch

class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1, padding_mode='replicate'), # 14x14 feature map dimension
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, padding_mode='replicate'), # 7x7 feature map dimension
            nn.ReLU(),
        )

        # y = xA^T+b
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        # extract feature maps
        x = self.conv(x)

        # flatten feature maps from (batch_size, feature_map_amount, feature_map_h, feature_map_w) to (batch_size, rest)
        x = x.view(x.size(0), -1)

        # calculate mu and standard deviation
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        # latent space -> feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
        )

        # 'undo' convolution from encoder
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),  # -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        # convert to feature map
        x = self.fc(z)
        # unflatten data
        x = x.view(-1, 128, 7, 7)

        # feature map -> image
        return self.deconv(x)

class VAE(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu # todo change

        recon_x = self.decoder(z)
        return recon_x, mu, logvar # todo change 0 to logvar

def vae_loss(recon_x, x, mu, logvar):
    #print("loss 1: ", x.max)
    recon_loss = f.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kld

    #print("loss 2: ", loss)
    return loss