import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from quantizer import ResidualVectorQuantizer

class EnCodec(nn.Module):
    def __init__(self, 
                 latent_dim=128,
                 channels=[128, 256, 384, 512],
                 kernel_size=7,
                 strides=[2, 2, 2, 2],
                 dilation_rates=[1, 3, 9],
                 num_codebooks=4,
                 vectors_per_codebook=1024):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Encoder(
            in_channels=1,
            latent_dim=latent_dim,
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rates=dilation_rates
        )
        
        # Vector Quantizer
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim,
            num_codebooks=num_codebooks,
            num_vectors=vectors_per_codebook
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=1,
            channels=channels[::-1],
            kernel_size=kernel_size,
            strides=strides[::-1],
            dilation_rates=dilation_rates
        )
        
        # Conditioning processor for f0 and phonemes
        self.f0_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Feed-forward layer to project conditioning to the right dimension
        # Keep the original expected dimension (64 + 128)
        self.conditioning_projection = nn.Linear(64 + 128, latent_dim)
    
    def encode(self, x):
        """Encode input audio to latent representation"""
        return self.encoder(x.unsqueeze(1))
    
    def decode(self, z, f0=None, phone_one_hot=None):
        """Decode latent representation with conditioning"""
        # Process conditioning signals
        if f0 is not None and phone_one_hot is not None:
            # Process f0
            f0 = f0.unsqueeze(1)  # [B, 1, T_f0]
            f0_features = self.f0_processor(f0)  # [B, 64, T_f0]
            
            # Prepare phoneme features
            B, T, C = phone_one_hot.shape
            
            # Pad phoneme one-hot vectors to 128 dimensions
            # This assumes the original model expects 128 phoneme dimensions
            expected_phoneme_dim = 128
            if C < expected_phoneme_dim:
                # Create padding
                padding = torch.zeros(B, T, expected_phoneme_dim - C, device=phone_one_hot.device)
                # Concatenate with original phoneme vectors
                padded_phone_one_hot = torch.cat([phone_one_hot, padding], dim=2)
            else:
                # If we already have enough dimensions, just use the original or truncate
                padded_phone_one_hot = phone_one_hot[:, :, :expected_phoneme_dim]
            
            # Downsample phoneme sequence to match latent frame rate
            phone_features = F.avg_pool1d(
                padded_phone_one_hot.transpose(1, 2),  # [B, C, T]
                kernel_size=z.shape[-1] // f0.shape[-1] if z.shape[-1] > f0.shape[-1] else 1,
                stride=z.shape[-1] // f0.shape[-1] if z.shape[-1] > f0.shape[-1] else 1
            )  # [B, C, T_latent]
            
            # Combine conditioning signals
            combined_features = torch.cat([
                f0_features,  # [B, 64, T_f0]
                phone_features  # [B, C, T_latent]
            ], dim=1)  # [B, 64+C, T_min]
            
            # Resize combined features to match z's time dimension using interpolation
            combined_features = F.interpolate(
                combined_features,
                size=z.shape[2],
                mode='linear',
                align_corners=False
            )
            
            # Project to latent dimension
            combined_features = combined_features.permute(0, 2, 1)  # [B, T_z, 64+C]
            condition = self.conditioning_projection(combined_features)  # [B, T_z, latent_dim]
            condition = condition.permute(0, 2, 1)  # [B, latent_dim, T_z]
            
            # Now z and condition have the same shape, so we can add them
            z = z + condition
        
        return self.decoder(z)
    
    def forward(self, x, f0=None, phone_one_hot=None):
        """Forward pass with optional conditioning"""
        # Encode
        z = self.encode(x)
        
        # Quantize
        q_z, indices, commitment_loss = self.quantizer(z)
        
        # Decode with conditioning
        x_hat = self.decode(q_z, f0, phone_one_hot)
        
        return x_hat.squeeze(1), commitment_loss, indices