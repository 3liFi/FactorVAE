# TODO never used, remove?
"""
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
"""

import torch
from training import VAELightning
import matplotlib.pyplot as plt
import torchvision
from vae import HyperParams, LATENT_DIM
from torch.utils.data import DataLoader

def sample_images(model_path="saved_models/vae_model_07_07_3.ckpt", hyper_params=HyperParams(), n=64):
    """
    This function uses a trained model to to sample n random images from the VAE. To do that, the latent vector is randomized.
    The images will be displayed in a plot.

    Args:
        model_path (String): The path to the trained model
        hyper_params (HyperParams): The hyperparameters to use in the VAEs convolution layers. Must match the parameters used when training the model at model_path.
        n (int): The number of images to sample
    """

    # Load model
    trainer = VAELightning(LATENT_DIM, hyper_params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    # Generate random latent vectors
    z = torch.randn(n, LATENT_DIM)
    # Feed through Decoder
    with torch.no_grad():
        samples = trainer.model.decoder(z)

    # Show images in plot
    grid = torchvision.utils.make_grid(samples, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def sample_latent_changes(dataset, model_path="vae_model_factor_chest.ckpt", hyper_params=HyperParams()):
    """
    This function visualizes the latent vector that the given model generates for the first image from the given dataset. For each latent dimension,
    it displays one row which makes isolated changes to that specific dimension. The value is changed from -3 to +3 in increments of 0.5. In the rightmost column,
    a difference map visualizes the changes between the steps.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to use for the visualization
        model_path (String): The path to the trained model
        hyper_params (HyperParams): The hyperparameters to use in the VAEs convolution layers. Must match the parameters used when training the model at model_path.
    """

    trainer = VAELightning(LATENT_DIM, hyper_params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x_batch, _ = next(iter(val_loader))

    x_batch_repeated = x_batch.repeat(13*LATENT_DIM, 1, 1, 1)

    x_batch = x_batch_repeated.to(trainer.device)

    # x_batch * 10 repeats the array 10 times
    mu, logvar = trainer.vae.encoder(x_batch)
    z = mu

    latent_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    for j in range(0, LATENT_DIM):
        for i in range(0, len(latent_vals)):
            print("changing!")
            z[j * len(latent_vals) + i][j] = latent_vals[i]

    reconstructed = trainer.vae.decoder(z)

    visualize_with_difference_maps(reconstructed, num_latents=32)


def visualize_with_difference_maps(reconstructed, num_latents=64, steps=13):
    """
    This function
    """

    images_per_row = steps + 1  # +1 for difference image
    rows = []

    threshold = 0.02  # You can tune this

    for i in range(num_latents):
        start = i * steps
        end = start + steps
        latent_imgs = reconstructed[start:end]  # (13, C, H, W)

        # 1. Compute std across 13 steps
        std = latent_imgs.std(dim=0)  # shape (C, H, W)

        # 2. Create mask of pixels that change significantly
        mask = (std > threshold).float()  # shape (C, H, W)

        # 3. Use the mean image just for visualization
        mean_img = latent_imgs.mean(dim=0)  # shape (C, H, W)

        # 4. Apply mask (only keep changed pixels)
        diff_img = mean_img * mask  # Changed pixels preserved, others black

        # 5. Add this difference image as 14th image
        row_imgs = torch.cat([latent_imgs, diff_img.unsqueeze(0)], dim=0)
        rows.append(row_imgs)

    # Combine all rows
    all_imgs = torch.cat(rows, dim=0)  # shape: (14 * num_latents, C, H, W)
    grid = torchvision.utils.make_grid(all_imgs, nrow=images_per_row, padding=1)

    # Show
    plt.figure(figsize=(images_per_row, num_latents))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.show()

def replicate_images(dataset, model_path="vae_model_factor_chest.ckpt", hyper_params=HyperParams()):
    """
    This function uses a trained model to replicate the first 10 images of the given dataset. It will display a grid of both the original
    and the generated images.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to replicate
        model_path (String): The path to the trained model
        hyper_params (HyperParams): The hyperparameters to use in the VAEs convolution layers. Must match the parameters used when training the model at model_path.
    """

    trainer = VAELightning(LATENT_DIM, hyper_params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    # Load the first 10 images of dataset
    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    x_batch, _ = next(iter(val_loader))

    # Pass through encoder to receive mu
    x_batch = x_batch.to(trainer.device)
    mu, _ = trainer.vae.encoder(x_batch)
    z = mu

    # Use mu directly as latent vector, without reparameterization, to generate the same image again
    reconstructed = trainer.vae.decoder(z)
    # Combine originals and generated images in grid
    combined_images = torch.cat([x_batch, reconstructed])

    # Show plot and save to PNG
    grid = torchvision.utils.make_grid(combined_images, nrow=5)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig("output.png")
    plt.show()