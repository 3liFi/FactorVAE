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

def sample_images(model_path="saved_models/vae_model_07_07_3.ckpt", n=64):
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

def sample_latent_changes(dataset, model_path="saved_models/vae_model_07_07_3.ckpt", latent_index=0):
    # we can pass default hyper params object here because it won't be used anyway
    trainer = VAELightning(LATENT_DIM, HyperParams())
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x_batch, _ = next(iter(val_loader))

    x_batch_repeated = x_batch.repeat(13*64, 1, 1, 1)

    x_batch = x_batch_repeated.to(trainer.device)

    # x_batch * 10 repeats the array 10 times
    mu, logvar = trainer.model.encoder(x_batch)
    z = mu

    latent_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    for j in range(0, 64):
        for i in range(0, len(latent_vals)):
            print("changing!")
            z[j * len(latent_vals) + i][j] = latent_vals[i]
        #pass
    #print(len(z))
    #print(z[0])

    print(z[0][latent_index])

    reconstructed = trainer.model.decoder(z)
    print("mu:", mu.min().item(), mu.max().item())
    print("logvar:", logvar.min().item(), logvar.max().item())
    print("Output range:", reconstructed.min().item(), reconstructed.max().item())

    #combined_images = torch.cat([reconstructed])

    grid = torchvision.utils.make_grid(reconstructed, nrow=13)
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.show()
    plt.savefig("output.png")

def replicate_images(dataset, model_path="saved_models/vae_model_07_07_3.ckpt"):
    # we can pass default hyper params object here because it won't be used anyway
    trainer = VAELightning(LATENT_DIM, HyperParams())
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    x_batch, _ = next(iter(val_loader))
    print("Input shape:", x_batch.shape)  # Should be [B, 3, H, W]
    print("Input min/max:", x_batch.min().item(), x_batch.max().item())  # Should be in [0, 1]

    x_batch = x_batch.to(trainer.device)
    mu, logvar = trainer.model.encoder(x_batch)
    z = mu
    print(len(z))
    print(z[0])
    reconstructed = trainer.model.decoder(z)
    print("mu:", mu.min().item(), mu.max().item())
    print("logvar:", logvar.min().item(), logvar.max().item())
    print("Output range:", reconstructed.min().item(), reconstructed.max().item())

    combined_images = torch.cat([x_batch, reconstructed])

    grid = torchvision.utils.make_grid(combined_images, nrow=5)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()
    plt.savefig("output.png")