import torch
import torch.nn as nn

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.norm2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation_rates=[1, 3, 9]):
        super().__init__()
        
        # Downsampling convolution
        self.conv_down = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride,
            padding=(kernel_size - 1) // 2
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Stack of residual units
        self.residual_stack = nn.ModuleList([
            ResidualUnit(out_channels, dilation=d) 
            for d in dilation_rates
        ])
    
    def forward(self, x):
        # Downsampling
        x = self.relu(self.norm(self.conv_down(x)))
        
        # Apply residual stack
        for layer in self.residual_stack:
            x = layer(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, channels=[128, 256, 384, 512], 
                 kernel_size=7, strides=[2, 2, 2, 2], dilation_rates=[1, 3, 9]):
        super().__init__()
        
        # Initial convolution to process the raw audio waveform
        self.conv_in = nn.Conv1d(
            in_channels, channels[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.norm_in = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU()
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownsamplingBlock(
                    channels[i], channels[i+1],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    dilation_rates=dilation_rates
                )
            )
        
        # Final projection to latent dimension
        self.proj = nn.Conv1d(channels[-1], latent_dim, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.norm_in(self.conv_in(x)))
        
        # Downsampling blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Project to latent dimension
        x = self.proj(x)
        
        return x