from torch.utils.data import DataLoader
from medmnist import PathMNIST
from vae import VAE, vae_loss
import pytorch_lightning as pl
import torch


class VAELightning(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = VAE(latent_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        loss = vae_loss(recon, x, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)



def train_model(transform):
    torch.set_float32_matmul_precision('high')
    train_dataset = PathMNIST(split='train', download=True, transform=transform)
    val_dataset = PathMNIST(split='val', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    vae_module = VAELightning(latent_dim=20)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        #accelerator='cpu'
    )
    trainer.fit(vae_module, train_loader, val_loader)

    #save model after training
    trainer.save_checkpoint("vae_model.ckpt")