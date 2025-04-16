import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio
from encodec import EnCodec
from losses import ReconstructionLoss, PhonemeAlignmentLoss
import time
import matplotlib.pyplot as plt
import numpy as np
import librosa
import io
from PIL import Image
import torchvision.transforms as transforms

class EnCodecLightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create EnCodec model
        self.model = EnCodec(
            latent_dim=config["model"]["latent_dim"],
            channels=config["model"]["channels"],
            kernel_size=config["model"]["kernel_size"],
            strides=config["model"]["strides"],
            dilation_rates=config["model"]["dilation_rates"],
            num_codebooks=config["quantizer"]["num_codebooks"],
            vectors_per_codebook=config["quantizer"]["vectors_per_codebook"]
        )
        
        # Loss functions
        self.reconstruction_loss_fn = ReconstructionLoss()
        
        # Initialize PhonemeAlignmentLoss if needed
        if config["losses"]["use_phoneme_alignment"]:
            from dataset import SingingVoiceDataset
            dataset = SingingVoiceDataset(rebuild_cache=False)
            self.num_phones = len(dataset.phone_map)
            self.phoneme_alignment_loss_fn = PhonemeAlignmentLoss(num_phones=self.num_phones)
        
        # For timing inference
        self.inference_times = []
    
    def forward(self, x, f0=None, phone_one_hot=None):
        return self.model(x, f0, phone_one_hot)
    
    def common_step(self, batch, batch_idx):
        x = batch["audio"]
        f0 = batch["f0"]
        phone_one_hot = batch["phone_one_hot"]
        phone_seq = batch["phone_seq"]
        
        # Forward pass
        start_time = time.time()
        x_hat, commitment_loss, indices = self(x, f0, phone_one_hot)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Calculate losses
        reconstruction_loss = self.reconstruction_loss_fn(x_hat, x)
        total_loss = reconstruction_loss * self.config["losses"]["reconstruction_weight"]
        
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "inference_time": inference_time
        }
        
        # Add commitment loss if enabled
        if self.config["quantizer"]["use_commitment_loss"]:
            commitment_loss_weighted = commitment_loss * self.config["quantizer"]["commitment_weight"]
            total_loss += commitment_loss_weighted
            loss_dict["commitment_loss"] = commitment_loss
        
        # Add phoneme alignment loss if enabled
        if self.config["losses"]["use_phoneme_alignment"]:
            phoneme_alignment_loss = self.phoneme_alignment_loss_fn(indices, phone_seq)
            phoneme_alignment_loss_weighted = phoneme_alignment_loss * self.config["losses"]["phoneme_alignment_weight"]
            total_loss += phoneme_alignment_loss_weighted
            loss_dict["phoneme_alignment_loss"] = phoneme_alignment_loss
        
        loss_dict["total_loss"] = total_loss
        
        return loss_dict, x, x_hat, indices, f0, phone_one_hot
    
    def training_step(self, batch, batch_idx):
        loss_dict, _, _, _, _, _ = self.common_step(batch, batch_idx)
        
        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_dict["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        loss_dict, x, x_hat, indices, f0, phone_one_hot = self.common_step(batch, batch_idx)
        
        # Log validation metrics
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log audio samples and visualizations
        current_epoch = self.current_epoch
        log_every_n_epochs = self.config["logging"]["log_audio_every_n_epochs"]
        
        if batch_idx == 0 and (current_epoch % log_every_n_epochs == 0 or current_epoch == 0):
            self.log_audio_samples(x, x_hat, batch_idx)
            self.log_spectrograms(x, x_hat, batch_idx)
        
        return loss_dict["total_loss"]
    
    def log_audio_samples(self, x, x_hat, batch_idx):
        num_samples = min(self.config["logging"]["num_audio_samples"], x.size(0))
        sample_rate = self.config["model"]["sample_rate"]
        
        for i in range(num_samples):
            # Log original audio
            self.logger.experiment.add_audio(
                f"original/sample_{i}",
                x[i].detach().cpu(),
                self.current_epoch,
                sample_rate=sample_rate
            )
            
            # Log reconstructed audio
            self.logger.experiment.add_audio(
                f"reconstructed/sample_{i}",
                x_hat[i].detach().cpu(),
                self.current_epoch,
                sample_rate=sample_rate
            )
    
    def log_spectrograms(self, x, x_hat, batch_idx):
        num_samples = min(self.config["logging"]["num_audio_samples"], x.size(0))
        sample_rate = self.config["model"]["sample_rate"]
        
        for i in range(num_samples):
            # Create figure with two spectrograms side by side
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original audio spectrogram
            x_np = x[i].detach().cpu().numpy()
            X_orig = librosa.amplitude_to_db(
                np.abs(librosa.stft(x_np)), ref=np.max)
            axs[0].imshow(X_orig, aspect='auto', origin='lower')
            axs[0].set_title('Original')
            axs[0].set_ylabel('Frequency')
            axs[0].set_xlabel('Time')
            
            # Reconstructed audio spectrogram
            x_hat_np = x_hat[i].detach().cpu().numpy()
            X_recon = librosa.amplitude_to_db(
                np.abs(librosa.stft(x_hat_np)), ref=np.max)
            axs[1].imshow(X_recon, aspect='auto', origin='lower')
            axs[1].set_title('Reconstructed')
            axs[1].set_ylabel('Frequency')
            axs[1].set_xlabel('Time')
            
            # Convert figure to image
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            
            # Convert PNG buffer to tensor
            img = transforms.ToTensor()(Image.open(buf))
            
            # Add to tensorboard
            self.logger.experiment.add_image(
                f"spectrogram_comparison/sample_{i}",
                img,
                self.current_epoch
            )
    
    def on_validation_epoch_end(self):
        # Log average inference time
        if len(self.inference_times) > 0:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            self.log("val/avg_inference_time", avg_inference_time)
            self.inference_times = []  # Reset for next epoch
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["training"]["learning_rate"]
        )
        
        return optimizer