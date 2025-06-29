from torch.utils.data import DataLoader

from vae import vae_loss
from vae import LATENT_DIM

def test_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(model.device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Validation loss: {avg_loss:.4f}")


def load_model(path="vae_model.ckpt"):
    model = VAELightning.load_from_checkpoint(path, latent_dim=LATENT_DIM)
    return model

# sample.py
import torch
from vae import VAE
from training import VAELightning
import matplotlib.pyplot as plt
import torchvision
from vae import HyperParams

def sample_images(model_path="vae_model.ckpt", n=64):
    # we can pass default hyper params object here because it won't be used anyway
    trainer = VAELightning(LATENT_DIM, HyperParams())
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    z = torch.randn(n, LATENT_DIM)
    with torch.no_grad():
        samples = trainer.model.decoder(z)

    grid = torchvision.utils.make_grid(samples, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def replicate_images(dataset, model_path="vae_model.ckpt"):
    # we can pass default hyper params object here because it won't be used anyway
    trainer = VAELightning(LATENT_DIM, HyperParams())
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(trainer.device)
    mu, logvar = trainer.model.encoder(x_batch)
    z = mu
    reconstructed = trainer.model.decoder(z)

    combined_images = torch.cat([x_batch, reconstructed])

    grid = torchvision.utils.make_grid(combined_images, nrow=5)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    plt.savefig("output.png")