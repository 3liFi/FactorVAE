from torch.utils.data import DataLoader

import sample
from gif_extractor import GifExtractor
from vae import VAE, vae_loss as vae_loss_fc, DiscriminatorModel
import pytorch_lightning as pl
import torch
from vae import LATENT_DIM
from vae import HyperParams
import torch.nn.functional as f
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from typing import cast


class VAELightning(pl.LightningModule):
    """
    An implementation of PyTorch Lightning. We use an adversarial approach documented in the paper FactorVAE (https://arxiv.org/pdf/1802.05983) that
    trains both a Discriminator model, and the traditional VAE. For that reason, we cannot use automatic_optimization.

    There is one optimizer for the Discriminator and one for the VAE. The Discriminator is trained on the output of the VAE, and the VAE uses the output of the Discriminator in the loss calculation.\
    """

    def __init__(self, latent_dim, params: HyperParams):
        super().__init__()
        self.automatic_optimization = False
        self.vae = VAE(latent_dim, params)
        self.discriminator = DiscriminatorModel(latent_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # We use two optimizers, one for the VAE and one for the Discriminator
        opts = self.optimizers()
        opt_vae = opts[0]
        opt_disc = opts[1]

        # Send data through VAE
        recon, mu, logvar, z = self.vae(x)

        if self.global_step % 100 == 0 and self.logger and hasattr(self.logger.experiment, "add_histogram"):
            self.logger.experiment.add_histogram("mu", mu, self.global_step)
            self.logger.experiment.add_histogram("logvar", logvar, self.global_step)

        # Calculate KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.log("KL Divergence", kld, logger=True)

        # Training process for Discriminator begins here
        z_detached = z.detach()

        # Permute latent vector randomly
        z_perm = torch.zeros_like(z_detached)
        for i in range(z.size(1)):
            z_perm[:, i] = z_detached[:, i][torch.randperm(z.size(0))]

        # Feed both the original latent vector (here: 'real') and permuted latent vector (here: 'fake') through the Discriminator
        real_logits = self.discriminator(z_detached)
        fake_logits = self.discriminator(z_perm)

        # 'Real' latent vector is class 1, 'Fake' latent vector is class 0
        real_labels = torch.ones(z.size(0), dtype=torch.long, device=self.device)
        fake_labels = torch.zeros(z.size(0), dtype=torch.long, device=self.device)

        # Calculate loss using cross_entropy
        d_loss_real = f.cross_entropy(real_logits, real_labels)
        d_loss_fake = f.cross_entropy(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # Optimize Discriminator
        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()

        self.log("disc_train_loss", d_loss, prog_bar=True, logger=True)

        # Training process for VAE begins here

        # Run Discriminator on real latent vector to see how well it is disentangled.
        # If not well disentangled, Discriminator will be able to confidently tell it is real
        # If well disentangled, Discriminator will have trouble telling real and permuted apart
        logits = self.discriminator(z)
        log_qz = logits[:, 1] - logits[:, 0]

        # Mean across entire batch is used as total_correlation value
        total_correlation = log_qz.mean()

        self.log("tc_estimate", total_correlation, logger=True)
        self.log("d_loss_real", d_loss_real, logger=True)
        self.log("d_loss_fake", d_loss_fake, logger=True)

        # Calculate VAE loss
        vae_loss, recon_loss = vae_loss_fc(recon, x, mu, logvar, total_correlation, self.current_epoch)

        self.log("vae_train_loss", vae_loss, prog_bar=True, logger=True)
        self.log("vae_recon_loss", recon_loss, prog_bar=True, logger=True)

        # Optimize VAE
        opt_vae.zero_grad()
        self.manual_backward(vae_loss)
        opt_vae.step()

    def configure_optimizers(self):
        opt_vae = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        return [opt_vae, opt_disc]


class EpochEndCallback(Callback):
    """
    This is a callback handler for VAELightning training. We use it to collect images at the end of every epoch.
    """

    def __init__(self, gif_extractor):
        self.gif_extractor = gif_extractor

    def on_train_epoch_end(self, trainer, pl_module):
        self.gif_extractor.append_figure(
            sample.replicate_images(self.gif_extractor.dataset, cast(VAELightning, pl_module), show=False, save=False))


def train_model(train_dataset, params: HyperParams):
    """
    This function trains our FactorVAE using VAELightning given the hyperparameters and a training dataset.

    Args:
        train_dataset (torch.utils.data.Dataset): the training dataset
        params (HyperParams): the hyperparameters that should be used for the VAEs convolution layers
    """

    torch.set_float32_matmul_precision('high')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, persistent_workers=True)

    vae_module = VAELightning(LATENT_DIM, params)
    gif_extractor = GifExtractor(train_dataset)

    logger = TensorBoardLogger(save_dir='lightning_logs', name='vae_3')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochEndCallback(gif_extractor)],
    )
    trainer.fit(vae_module, train_loader)

    # export gif from training
    gif_extractor.export_gif("gifs/test.gif")

    # save model after training
    trainer.save_checkpoint("vae_model_factor_chest.ckpt")
