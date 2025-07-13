from typing import NamedTuple

import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import torch
import math

# FEATURE_MAP_H = 7
# FEATURE_MAP_W = 7
LATENT_DIM = 16
SOURCE_IMAGE_DIM = 28


class HyperParams(NamedTuple):
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1


"""
This function ignores dilation for now.
"""


def calc_fmap_size(source_dim: int, kernel_size: int, stride: int, padding: int) -> int:
    return math.floor((source_dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


def calc_outer_padding_based_on_desired_output_dims(source_dim: int, target_dim: int, kernel_size: int, stride: int,
                                                    padding: int) -> int:
    curr_out_dim = (source_dim - 1) * stride - 2 * padding + kernel_size  # + outer_padding
    return target_dim - curr_out_dim



class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=0, stride=1)       # 24 x 24
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=0, stride=1)      # 20 x 20
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=0, stride=1)     # 16 x 16
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2, stride=2)    # 8 x 8
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv5 = nn.Conv2d(64, 256, 5, padding=2, stride=2)  # 4 x 4
        self.bn5 = nn.BatchNorm2d(256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn6 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc_mean = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        out = out.view(batch_size, -1)
        out = self.relu(self.bn6(self.fc1(out)))
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, 4 * 4 * 256)
        self.bn0 = nn.BatchNorm1d(4 * 4 * 256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.deconv0 = nn.ConvTranspose2d(256, 64, 6, padding=2, stride=2)  # 8 x 8
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 6, padding=2, stride=2) # 16 x 16
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, padding=0, stride=1)  # 20 x 20
        self.bn3 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 5, padding=0, stride=1)   # 24 x 24
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 5, padding=0, stride=1)    # 28 x 28
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn0(self.fc1(x)))
        x = x.view(-1, 256, 4, 4)
        x = self.relu(self.bn1(self.deconv0(x)))
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))
        # print(f"Decoder output shape: {x.shape}")
        return x


class DiscriminatorModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(1000, 1),
        )

    def forward(self, z):
        return self.discriminator(z)

class VAE(nn.Module):

    def __init__(self, latent_dim, params: HyperParams):
        super().__init__()

        first_conv_trans_2d_layer_dim = calc_fmap_size(
            SOURCE_IMAGE_DIM, params.kernel_size, params.stride, params.padding)

        # apply function twice for actual feature map dim because we have two Conv2d layers
        feature_map_dim = calc_fmap_size(
            first_conv_trans_2d_layer_dim, params.kernel_size, params.stride, params.padding
        )

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # sample from a normal distribution with mean 0 and variance 1
        eps = torch.randn_like(std)

        # offset mu by small value within standard deviation
        return mu + eps * std
        # todo re-enable random picks eventually
        #return mu


def vae_loss(recon_x, x, mu, logvar, total_correlation, current_epoch):
    # print("loss 1: ", x.max)
    #recon_loss = f.binary_cross_entropy(recon_x, x, reduction='sum')
    # recon_loss = f.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    recon_loss = f.mse_loss(recon_x, x, reduction="sum")
    #recon_loss = f.l1_loss(recon_x, x, reduction="sum")
    #logvar = torch.clamp(logvar, min=-10, max=10)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # for 30 epochs, this goes from 0.002 to 0.06
    #beta = min(1, (current_epoch - 10) / 400)
    #if current_epoch <= 10:
    #  beta = 0.001
    beta = 0.001
    #beta = min(1.0, current_epoch / 200 * 0.05)
    #loss = recon_loss + beta * kld
    gamma = 10.0
    loss = recon_loss + kld * beta + gamma * total_correlation
    # print("loss 2: ", loss)
    return loss