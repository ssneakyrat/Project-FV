import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, dim, num_vectors=1024):
        super().__init__()
        
        self.dim = dim
        self.num_vectors = num_vectors
        
        # Initialize the codebook
        self.codebook = nn.Parameter(torch.randn(num_vectors, dim))
        self.codebook.data.uniform_(-1/num_vectors, 1/num_vectors)
    
    def forward(self, z):
        """
        z: [B, C, T] tensor of latent representations
        """
        batch_size, channels, time = z.shape
        
        # Reshape z to [B*T, C]
        z_flat = z.permute(0, 2, 1).reshape(-1, channels)
        
        # Calculate distances between latent vectors and codebook
        distances = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.codebook ** 2, dim=1) - \
                   2 * torch.matmul(z_flat, self.codebook.t())
        
        # Find nearest codebook vector for each latent vector
        min_indices = torch.argmin(distances, dim=1)
        
        # Get the corresponding quantized vectors
        z_q_flat = self.codebook[min_indices]
        
        # Reshape back to [B, C, T]
        z_q = z_q_flat.view(batch_size, time, channels).permute(0, 2, 1)
        
        # Calculate commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, min_indices.view(batch_size, time), commitment_loss

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_codebooks=4, num_vectors=1024):
        super().__init__()
        
        self.dim = dim
        self.num_codebooks = num_codebooks
        
        # Create multiple VQ layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(dim, num_vectors) for _ in range(num_codebooks)
        ])
    
    def forward(self, z):
        """
        z: [B, C, T] tensor of latent representations
        """
        z_q = torch.zeros_like(z)
        residual = z
        
        indices_list = []
        commitment_loss_total = 0
        
        # Apply residual vector quantization
        for i, vq_layer in enumerate(self.vq_layers):
            z_q_i, indices_i, commitment_loss_i = vq_layer(residual)
            
            # Add quantized vector to the output
            z_q = z_q + z_q_i
            
            # Update residual
            residual = residual - z_q_i.detach()  # detach to prevent gradients flowing through residual path
            
            # Collect indices and commitment loss
            indices_list.append(indices_i)
            commitment_loss_total += commitment_loss_i
        
        # Average commitment loss across codebooks
        commitment_loss_total /= len(self.vq_layers)
        
        # Stack indices for all codebooks
        indices = torch.stack(indices_list, dim=1)  # [B, num_codebooks, T]
        
        return z_q, indices, commitment_loss_total