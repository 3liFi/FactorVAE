from typing import NamedTuple

import torch.nn as nn
import torch.nn.functional as f
import torch
import math

# FEATURE_MAP_H = 7
# FEATURE_MAP_W = 7
LATENT_DIM = 256
SOURCE_IMAGE_DIM = 28


class HyperParams(NamedTuple):
    kernel_size: int = 3
    stride: int = 2
    padding: int = 0

# Default hyperparameters for the VAE
params = HyperParams(kernel_size=3, stride=1, padding=0)


"""
This function ignores dilation for now.
"""


def calc_fmap_size(source_dim: int, kernel_size: int, stride: int, padding: int) -> int:
    return math.floor((source_dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


def calc_outer_padding_based_on_desired_output_dims(source_dim: int, target_dim: int, kernel_size: int, stride: int,
                                                    padding: int) -> int:
    curr_out_dim = (source_dim - 1) * stride - 2 * padding + kernel_size  # + outer_padding
    return target_dim - curr_out_dim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=0, stride=1)       # 24 x 24
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0, stride=1)      # 20 x 20
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=0, stride=1)     # 16 x 16
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)    # 8 x 8
        self.bn4 = nn.BatchNorm2d(256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.bn5 = nn.BatchNorm1d(1024, momentum=0.9)
        self.fc_mean = nn.Linear(1024, LATENT_DIM)
        self.fc_logvar = nn.Linear(1024, LATENT_DIM)


    def forward(self, x):
        batch_size = x.size()[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = out.view(batch_size, -1)
        out = self.relu(self.bn5(self.fc1(out)))
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(LATENT_DIM, 8 * 8 * 256)
        self.bn1 = nn.BatchNorm1d(8 * 8 * 256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 6, padding=2, stride=2) # 16 x 16
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, padding=0, stride=1)  # 20 x 20
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, padding=0, stride=1)   # 24 x 24
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, padding=0, stride=1)    # 28 x 28
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, 8, 8)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))
        # print(f"Decoder output shape: {x.shape}")
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=0, stride=1)       # 24 x 24
        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0, stride=1)      # 20 x 20
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=0, stride=1)     # 16 x 16
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)    # 8 x 8
        self.bn4 = nn.BatchNorm2d(256, momentum=0.9)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.9)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x1 = x
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x, x1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VAE(nn.Module):

    def __init__(self, latent_dim, params: HyperParams):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.discriminator = Discriminator()
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        # self.discriminator.apply(weights_init)


    def forward(self, x):
        bs = x.size()[0]
        z_mean, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()

        # sampling epsilon from normal distribution
        epsilon = torch.randn(bs, LATENT_DIM).to(device)
        z = z_mean + std * epsilon
        x_tilda = self.decoder(z)

        return x_tilda, z_mean, z_logvar

    def reparameterize(self, mu, logvar):




        # std = torch.exp(0.5 * logvar)
        # sample from a normal distribution with mean 0 and variance 1
        # eps = torch.randn_like(std)

        # offset mu by small value within standard deviation
        # return mu + eps * std
        # todo re-enable random picks eventually
        return mu


def vae_loss(recon_x, x, mu, logvar):
    # print("loss 1: ", x.max)
    recon_loss = f.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kld * 0.0001

    # print("loss 2: ", loss)
    return loss
