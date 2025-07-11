from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
from training import train_model, optimize_hyper_params_with_optuna
from sample import sample_images, replicate_images
from vae import params
import argparse

data_flag = 'pathmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(__import__('medmnist'), info['python_class'])

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = DataClass(split='train', transform=transform, download=download, size=28, mmap_mode='r')
val_dataset = DataClass(split='val', transform=transform, download=download, size=28, mmap_mode='r')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'random', 'replicate', 'optimize'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(transform, params)
    elif args.mode == 'random':
        sample_images()
    elif args.mode == 'replicate':
        replicate_images(val_dataset)
    elif args.mode == 'optimize':
        optimize_hyper_params_with_optuna()

