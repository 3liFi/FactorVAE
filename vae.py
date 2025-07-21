from typing import NamedTuple

import torch.nn as nn
import torch.nn.functional as f
import torch
import math

# FEATURE_MAP_H = 7
# FEATURE_MAP_W = 7
LATENT_DIM = 8
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

    def __init__(self, latent_dim, params: HyperParams, feature_map_dim: int):
        super().__init__()

        self.params = params

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, params.kernel_size, params.stride, params.padding, padding_mode='replicate'),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, params.kernel_size, params.stride, params.padding, padding_mode='replicate'),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, params.kernel_size, 1, params.padding, padding_mode='replicate'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(512 * feature_map_dim * feature_map_dim, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(64, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512 * feature_map_dim * feature_map_dim, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(64, latent_dim),
            #nn.Tanh(),
        )

    def forward(self, x):
        # extract feature maps
        x = self.conv(x)

        # flatten feature maps from (batch_size, feature_map_amount, feature_map_h, feature_map_w) to (batch_size, rest)
        x = x.view(x.size(0), -1)

        # calculate mu and standard deviation
        return self.fc_mu(x), self.fc_logvar(x)# * 4


class Decoder(nn.Module):

    def __init__(self, latent_dim, params: HyperParams, feature_map_dim: int, first_conv_trans_2d_layer_dim: int):
        super().__init__()

        self.params = params
        self.feature_map_dim = feature_map_dim

        # latent space -> feature map
        self.fc = nn.Sequential(
            #nn.Linear(latent_dim, 64),
            #nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 1024 * self.feature_map_dim * self.feature_map_dim),
            nn.LeakyReLU(0.2, inplace=False),
            #nn.Linear(1024, 2048 * self.feature_map_dim * self.feature_map_dim),
            #nn.LeakyReLU(0.2, inplace=False),
        )

        # todo calculate outer padding based on desired output size
        # 'undo' convolution from encoder
        self.deconv = nn.Sequential(
            # this layer should keep feature_map_dim resolution, so stride needs to be adjusted
            #nn.ConvTranspose2d(
            #    2048, 1024, params.kernel_size, stride=1, padding=params.padding,
            #    output_padding=calc_outer_padding_based_on_desired_output_dims(
            #        feature_map_dim, feature_map_dim, params.kernel_size, 1, params.padding
            #    )
            #),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=False),
            # this layer should keep feature_map_dim resolution, so stride needs to be adjusted
            nn.ConvTranspose2d(
                1024, 512, params.kernel_size, stride=1, padding=params.padding,
                output_padding=calc_outer_padding_based_on_desired_output_dims(
                    feature_map_dim, feature_map_dim, params.kernel_size, 1, params.padding
                )
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            # this layer should keep feature_map_dim resolution, so stride needs to be adjusted
            nn.ConvTranspose2d(
                512, 256, params.kernel_size, stride=1, padding=params.padding,
                output_padding=calc_outer_padding_based_on_desired_output_dims(
                    feature_map_dim, feature_map_dim, params.kernel_size, 1, params.padding
                )
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(
                256, 128, params.kernel_size, stride=params.stride, padding=params.padding,
                output_padding=calc_outer_padding_based_on_desired_output_dims(
                    feature_map_dim, first_conv_trans_2d_layer_dim, params.kernel_size, params.stride, params.padding
                )
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(128, 1, params.kernel_size, stride=params.stride, padding=params.padding,
                               output_padding=calc_outer_padding_based_on_desired_output_dims(
                                   first_conv_trans_2d_layer_dim, SOURCE_IMAGE_DIM, params.kernel_size, params.stride,
                                   params.padding
                               )),  # -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        # convert to feature map
        x = self.fc(z)
        # unflatten data
        x = x.view(-1, 1024, self.feature_map_dim, self.feature_map_dim)

        # feature map -> image
        return self.deconv(x)


class DiscriminatorModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1000, 2)
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

        self.encoder = Encoder(latent_dim, params, feature_map_dim)
        self.decoder = Decoder(latent_dim, params, feature_map_dim, first_conv_trans_2d_layer_dim)

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
    #beta = 0.5
    target_beta = 0.7
    beta = min(target_beta, max(0.001, (current_epoch / 30) * target_beta))
    #beta = min(1.0, current_epoch / 200 * 0.05)
    #loss = recon_loss + beta * kld

    gamma = 0

    #if current_epoch < 5:
        #gamma = 0.0
    loss = recon_loss + kld * beta + gamma * total_correlation
    # print("loss 2: ", loss)
    return loss, recon_loss