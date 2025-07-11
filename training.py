from torch.utils.data import DataLoader
from medmnist import PathMNIST
from vae import VAE, vae_loss as vae_loss_fc, DiscriminatorModel
import pytorch_lightning as pl
import torch
from vae import LATENT_DIM
from vae import HyperParams
from torchvision import transforms
import optuna
import optuna.visualization as vis
from image_similarity import compute_ssim
import joblib
import torch.nn.functional as f

#  Best is trial 2 with value: 0.48198455891420766. {'kernel_size': 3, 'stride': 2, 'padding': 0}

class VAELightning(pl.LightningModule):
    def __init__(self, latent_dim, params: HyperParams):
        super().__init__()
        self.automatic_optimization = False
        self.vae = VAE(latent_dim, params)
        self.discriminator = DiscriminatorModel(latent_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_vae, opt_disc = self.optimizers()

        # forward pass
        recon, mu, logvar, z = self.vae(x)

        # train discriminator
        with torch.no_grad():
            z_detached = z.detach()

        z_perm = z_detached[torch.randperm(z_detached.size(0))]

        real_logits = self.discriminator(z_detached)
        fake_logits = self.discriminator(z_perm)

        real_labels = torch.ones(z.size(0), dtype=torch.long, device=self.device)
        fake_labels = torch.zeros(z.size(0), dtype=torch.long, device=self.device)

        d_loss_real = f.cross_entropy(real_logits, real_labels)
        d_loss_fake = f.cross_entropy(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()

        self.log("disc_train_loss", d_loss, prog_bar=True)

        # train VAE
        logits = self.discriminator(z)
        log_qz = logits[:, 0] - logits[:, 1]
        total_correlation = log_qz.mean()

        vae_loss = vae_loss_fc(recon, x, mu, logvar, total_correlation, self.current_epoch)

        self.log("vae_train_loss", vae_loss, prog_bar=True)

        opt_vae.zero_grad()
        self.manual_backward(vae_loss)
        opt_vae.step()
        # VAE optimizer
        """if optimizer_idx == 0:
            logits = self.discriminator(z)
            log_qz = logits[:, 0] - logits[:, 1]
            total_correlation = log_qz.mean()

            loss = vae_loss(recon, x, mu, logvar, total_correlation, self.current_epoch)
            self.log("vae_train_loss", loss)

            return loss
        # Discriminator optimizer
        elif optimizer_idx == 1:
            z_detached = z.detach()
            real_logits = self.discriminator(z_detached)

            # Permuted z (simulate product of marginals)
            z_perm = z_detached[torch.randperm(z_detached.size(0))]
            fake_logits = self.discriminator(z_perm)

            real_labels = torch.ones(z.size(0), dtype=torch.long, device=self.device)
            fake_labels = torch.zeros(z.size(0), dtype=torch.long, device=self.device)

            d_loss_real = f.cross_entropy(real_logits, real_labels)
            d_loss_fake = f.cross_entropy(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            self.log("disc_train_loss", d_loss)
            return d_loss"""

        """x, _ = batch
        recon, mu, logvar = self.vae(x)
        loss = vae_loss(recon, x, mu, logvar, self.current_epoch)
        self.log("train_loss", loss)
        return loss"""

    def configure_optimizers(self):
        opt_vae = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        return [opt_vae, opt_disc]

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

    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    stride = trial.suggest_int("stride", 1, 2)
    padding = trial.suggest_categorical("padding", [0, kernel_size // 2])
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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    vae_module = VAELightning(LATENT_DIM, params)
    #vae_module = VAELightning.load_from_checkpoint("vae_model_large_100_epochs_annealing.ckpt", latent_dim=LATENT_DIM, params=params)

    trainer = pl.Trainer(
        max_epochs=75,
       # resume_from_checkpoint="vae_model.ckpt",
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        #accelerator='cpu'
    )
    trainer.fit(vae_module, train_loader, val_loader)

    #save model after training
    trainer.save_checkpoint("vae_model_best_32_latent_factor.ckpt")
