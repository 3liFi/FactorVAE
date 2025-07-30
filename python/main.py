# MIT License
# Copyright (c) 2025 Elias Fiedler, Fynn Becker, Patrick Reidelbach
# See LICENSE file in the project root for full license information.

from medmnist import INFO
from torchvision import transforms
from model.training import train_model
from sample import sample_images, replicate_images, sample_latent_changes, load_trainer_from_model_path
import argparse
from model.vae import HyperParams

data_flag = 'chestmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(__import__('medmnist'), info['python_class'])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor()
])

train_dataset = DataClass(split='train', transform=transform, download=download)
val_dataset = DataClass(split='val', transform=transform, download=download)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'random', 'replicate', 'latent'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(train_dataset, HyperParams())
    elif args.mode == 'random':
        sample_images()
    elif args.mode == 'replicate':
        replicate_images(val_dataset, load_trainer_from_model_path())
    elif args.mode == 'latent':
        sample_latent_changes(val_dataset, load_trainer_from_model_path())
