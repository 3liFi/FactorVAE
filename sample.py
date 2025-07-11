from torch.utils.data import DataLoader

from vae import vae_loss
from vae import LATENT_DIM
from vae import params

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

default_model_path = "checkpoints_256/vae-epoch=94-train_loss=14840.05.ckpt"

# sample.py
import torch
from vae import VAE
from training import VAELightning
import matplotlib.pyplot as plt
import torchvision

def sample_images(model_path=default_model_path, n=64):
    trainer = VAELightning(LATENT_DIM, params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    z = torch.randn(n, LATENT_DIM)
    with torch.no_grad():
        samples = trainer.model.decoder(z)

    # Assuming samples are in the range [0, 1]
    # samples = (samples + 1) / 2  # Normalize to [0, 1]
    # samples = samples.clamp(0, 1)  # Ensure values are within [0, 1]
    # samples = samples.cpu()
    # samples = samples.view(n, 1, 64, 64)  # Reshape if needed
    # samples = samples[:64]  # Limit to 64 samples for visualization
    # samples = samples.permute(0, 2, 3, 1)  # Change to (N, H, W, C) for matplotlib
    # samples = samples.numpy()  # Convert to numpy for visualization
    # samples = samples * 255  # Scale to [0, 255] for visualization
    # samples = samples.astype('uint8')  # Convert to uint8 for image display
    # samples = samples.reshape(n, 64, 64, 1)  # Reshape if needed
    # samples = samples.squeeze()  # Remove channel dimension if grayscale
    # samples = samples[:64]  # Limit to 64 samples for visualization

    grid = torchvision.utils.make_grid(samples, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def replicate_images(dataset, model_path=default_model_path):
    trainer = VAELightning(LATENT_DIM, params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(trainer.device)
    mu, logvar = trainer.model.encoder(x_batch)
    z = mu
    reconstructed = trainer.model.decoder(z)
    # # Ensure reconstructed images are in the range [0, 1]
    # reconstructed = (reconstructed + 1) / 2  # Normalize to [0, 1]

    combined_images = torch.cat([x_batch, reconstructed])

    grid = torchvision.utils.make_grid(combined_images, nrow=5)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
