from torch.utils.data import DataLoader
from medmnist import PathMNIST
from vae import VAE, vae_loss
import pytorch_lightning as pl
import torch
from vae import LATENT_DIM
from vae import HyperParams
from torchvision import transforms
import optuna
import optuna.visualization as vis
from image_similarity import compute_ssim
import joblib

#  Best is trial 2 with value: 0.48198455891420766. {'kernel_size': 3, 'stride': 2, 'padding': 0}

class VAELightning(pl.LightningModule):
    def __init__(self, latent_dim, params: HyperParams):
        super().__init__()
        self.model = VAE(latent_dim, params)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        loss = vae_loss(recon, x, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

def get_model_similarity_score(dataset, vae_lightning: VAELightning) -> int:
    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    x_batch, _ = next(iter(val_loader))

    #x_batch = x_batch.to(trainer.device)
    mu, logvar = vae_lightning.model.encoder(x_batch)
    z = mu
    reconstructed = vae_lightning.model.decoder(z)

    total_ssim_score = 0
    for i in range(0, len(x_batch)):
        total_ssim_score += compute_ssim(reconstructed[i], x_batch[i])

    return total_ssim_score / len(x_batch)

def optimize_hyper_params_with_optuna():
    study = optuna.create_study(direction="maximize")  # or "maximize" if you're maximizing SSIM etc.
    study.optimize(objective, n_trials=50)

    # Plot how the score improved over time
    vis.plot_optimization_history(study).show()

    # See which parameters mattered most
    vis.plot_param_importances(study).show()

    # Parallel coordinate plot (explores interactions)
    vis.plot_parallel_coordinate(study).show()

    joblib.dump(study, "vae_study_0.pkl")

def objective(trial):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor()
    ])

    train_dataset = PathMNIST(split='train', download=True, transform=transform)
    val_dataset = PathMNIST(split='val', download=True, transform=transform)

    # Hyperparameters to optimize
    kernel_size = trial.suggest_int("kernel_size", 3, 7, 2)
    stride = trial.suggest_int("stride", 1, 2)
    padding = trial.suggest_int("padding", 0, 4, 2)
    LATENT_DIM = trial.suggest_int("latent_dim", 10, 100, 10)
    epochs = trial.suggest_int("epochs", 5, 35, 5)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = VAELightning(LATENT_DIM, HyperParams(kernel_size, stride, padding))

    trainer.fit(model, train_loader, val_loader)

    # Use visual comparison metric or val loss
    #return trainer.callback_metrics["train_loss"].item()
    return get_model_similarity_score(val_dataset, model)

def train_model(transform, params: HyperParams):
    torch.set_float32_matmul_precision('high')
    train_dataset = PathMNIST(split='train', download=True, transform=transform)
    val_dataset = PathMNIST(split='val', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    vae_module = VAELightning(LATENT_DIM, params)

    trainer = pl.Trainer(
        max_epochs=100,
        # save
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints_256_chest',
            filename='vae-{epoch:02d}',
            save_top_k=-1,
            every_n_epochs=5
        )],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        #accelerator='cpu'
    )
    trainer.fit(vae_module, train_loader, val_loader)

    #save model after training
    trainer.save_checkpoint("vae_model.ckpt")