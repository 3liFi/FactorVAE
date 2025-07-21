import imageio
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io
import torch
import torchvision
import matplotlib.pyplot as plt

class GifExtractor:

    def __init__(self, dataset):
        self.images = []
        self.dataset = dataset

    def append_image(self, image):
        self.images.append(image)

    def append_figure(self, figure):
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert("RGB")

        self.images.append(img)

    def auto_sample_replicate_image_figure(self, trainer):
        val_loader = DataLoader(self.dataset, batch_size=10, shuffle=False)
        x_batch, _ = next(iter(val_loader))
        x_batch = x_batch.to(trainer.device)

        mu, logvar = trainer.vae.encoder(x_batch)
        z = mu

        reconstructed = trainer.vae.decoder(z)

        combined_images = torch.cat([x_batch, reconstructed])

        grid = torchvision.utils.make_grid(combined_images, nrow=5)
        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.axis('off')

        self.append_figure(fig)

        plt.close(fig)

    def auto_sample_image(self, trainer):
        val_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        x_batch, _ = next(iter(val_loader))

        x_batch = x_batch.to(trainer.device)
        mu, logvar = trainer.vae.encoder(x_batch)
        z = mu

        reconstructed = trainer.vae.decoder(z)
        img = reconstructed[0]
        img = img.permute(1, 2, 0).cpu().detach().numpy()

        img = np.clip(img, 0, 1)

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

        img = (img * 255).astype(np.uint8)

        self.images.append(img)

    def export_gif(self, path):
        imageio.mimsave(path, self.images)