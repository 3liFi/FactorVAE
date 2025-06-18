from medmnist import INFO
from torchvision import transforms
from training import train_model
from sample import sample_images
import argparse

data_flag = 'pathmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(__import__('medmnist'), info['python_class'])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop((28,28)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=transform, download=download)
val_dataset = DataClass(split='val', transform=transform, download=download)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'sample'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(transform)
    elif args.mode == 'sample':
        sample_images()


