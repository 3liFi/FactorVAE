from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
from training import train_model, optimize_hyper_params_with_optuna
from sample import sample_images, replicate_images
import argparse
from vae import HyperParams

data_flag = 'pathmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(__import__('medmnist'), info['python_class'])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop((28,28)),
    transforms.ToTensor()
])

train_dataset = DataClass(split='train', transform=transform, download=download)
val_dataset = DataClass(split='val', transform=transform, download=download)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'random', 'replicate', 'optimize'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(transform, HyperParams())
    elif args.mode == 'random':
        sample_images()
    elif args.mode == 'replicate':
        replicate_images(val_dataset)
    elif args.mode == 'optimize':
        optimize_hyper_params_with_optuna()