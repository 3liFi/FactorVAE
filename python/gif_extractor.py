# MIT License
# Copyright (c) 2025 Elias Fiedler, Fynn Becker, Patrick Reidelbach
# See LICENSE file in the project root for full license information.

import imageio
from PIL import Image
import io


class GifExtractor:
    """
    A small helper class that collects images and is able to export them to a GIF file. Used to visualize changes in the models reconstruction ability
    as training progresses.
    """

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

    def export_gif(self, path):
        imageio.mimsave(path, self.images)
