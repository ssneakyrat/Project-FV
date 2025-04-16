import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_hat, x):
        """
        Simple L1 loss for audio reconstruction
        """
        return F.l1_loss(x_hat, x)

class PhonemeAlignmentLoss(nn.Module):
    def __init__(self, num_phones):
        super().__init__()
        self.num_phones = num_phones
    
    def forward(self, indices, phone_seq):
        """
        Loss to align quantized tokens with phone sequence
        
        Args:
            indices: [B, num_codebooks, T_latent] tensor of codebook indices
            phone_seq: [B, T_audio] tensor of phone indices
        """
        batch_size, num_codebooks, time_latent = indices.shape
        batch_size, time_audio = phone_seq.shape
        
        # Use only the first codebook for alignment (most significant)
        first_indices = indices[:, 0, :]  # [B, T_latent]
        
        # Downsample phone sequence to match latent framerate
        ratio = time_audio // time_latent
        if ratio > 1:
            phone_seq_downsampled = F.avg_pool1d(
                F.one_hot(phone_seq, self.num_phones).float().transpose(1, 2),
                kernel_size=ratio,
                stride=ratio
            ).argmax(dim=1)  # [B, T_latent]
        else:
            # Upsample the indices if needed
            first_indices = F.interpolate(
                first_indices.unsqueeze(1).float(),
                size=time_audio,
                mode='nearest'
            ).squeeze(1).long()
            phone_seq_downsampled = phone_seq
        
        # Create mask for padding (where phone_seq_downsampled == 0)
        mask = (phone_seq_downsampled > 0).float()
        
        # Cross-entropy loss for phone prediction
        loss = F.cross_entropy(
            first_indices.reshape(-1, 1).float(),  # Convert input to float
            phone_seq_downsampled.reshape(-1),
            reduction='none'
        )
        
        # Apply mask and average
        loss = (loss * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)  # Changed from view to reshape
        
        return loss