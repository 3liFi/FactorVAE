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


def sample_latent_changes(dataset, model_path="vae_model_factor_chest.ckpt", hyper_params=HyperParams()):
    """
    This function visualizes the latent vector that the given model generates for the first image from the given dataset. For each latent dimension,
    it displays one row which makes isolated changes to that specific dimension. The value is changed from -3 to +3 in increments of 0.5. In the rightmost column,
    a difference map visualizes the changes between the steps.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to use for the visualization
        model_path (String): The path to the trained model
        hyper_params (HyperParams): The hyperparameters to use in the VAEs convolution layers. Must match the parameters used when training the model at model_path.

    Notes:
        While this implementation works, it is not the most efficient. Ideally, the encoder would only be queried with one image. That output should be cloned several times
        and edited by inserting the latent_vals as one dimension of the latent vector tensor. For our purposes, the current implementation will do as it is only used for manual
        evaluation right now. It is not advised to use this function in any sort of automatic evaluation behavior that happens frequently during e.g. training.
    """

    # Load model
    trainer = VAELightning(LATENT_DIM, hyper_params)
    trainer.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.eval()

    # Load data
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x_batch, _ = next(iter(val_loader))

    # Repeat 13 times, as there are 13 values will insert for each dimension
    x_batch_repeated = x_batch.repeat(13 * LATENT_DIM, 1, 1, 1)
    x_batch = x_batch_repeated.to(trainer.device)

    # Encode batch to receive latent vector
    mu, logvar = trainer.vae.encoder(x_batch)
    z = mu

    # Alter each latent vector by switching dimension j to the values from latent_vals
    latent_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    for j in range(0, LATENT_DIM):
        for i in range(0, len(latent_vals)):
            z[j * len(latent_vals) + i][j] = latent_vals[i]

    # Use Decoder to generate images
    reconstructed = trainer.vae.decoder(z)

    # Add difference maps
    visualize_with_difference_maps(reconstructed, group_size=len(latent_vals))


def visualize_with_difference_maps(images, group_size=13, threshold=0.02):
    """
    This function accepts an array of images and visualizes the difference between every group by generating a 'difference map', which is an image
    that only displays pixels that have changed more than a certain amount between the pictures. The result is displayed in a plot where each group of images
    is shown in a separate row, together with the difference map at the end of the row.

    Args:
        images (list[torch.Tensor]): The array of images
        group_size (int): The number of images that the difference should be visualized between.
        threshold (float): The threshold above which the difference between pixels must be. Can be tuned to reflect different use cases (e.g., detecting any changes vs. detecting severe changes)
    """

    images_per_row = group_size + 1
    rows = []

    for i in range(LATENT_DIM):
        start = i * group_size
        end = start + group_size

        # Extract relevant images from reconstruction data
        latent_imgs = images[start:end]

        # Compute standard deviation
        std = latent_imgs.std(dim=0)

        # Any pixels that changed more than the threshold will be part of the mask
        mask = (std > threshold).float()

        # Apply mask to mean image for visualization
        mean_img = latent_imgs.mean(dim=0)
        diff_img = mean_img * mask

        # Add as last image in row
        row_imgs = torch.cat([latent_imgs, diff_img.unsqueeze(0)], dim=0)
        rows.append(row_imgs)

    # Combine all rows
    all_imgs = torch.cat(rows, dim=0)
    grid = torchvision.utils.make_grid(all_imgs, nrow=images_per_row, padding=1)

    # Show plot
    plt.figure(figsize=(images_per_row, LATENT_DIM))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.show()
