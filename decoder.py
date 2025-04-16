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

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation_rates=[1, 3, 9]):
        super().__init__()
        
        # Upsampling convolution (transposed convolution)
        self.conv_up = nn.ConvTranspose1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride,
            padding=(kernel_size - 1) // 2,
            output_padding=stride - 1
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Stack of residual units
        self.residual_stack = nn.ModuleList([
            ResidualUnit(out_channels, dilation=d) 
            for d in dilation_rates
        ])
    
    def forward(self, x):
        # Upsampling
        x = self.relu(self.norm(self.conv_up(x)))
        
        # Apply residual stack
        for layer in self.residual_stack:
            x = layer(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1, channels=[512, 384, 256, 128], 
                 kernel_size=7, strides=[2, 2, 2, 2], dilation_rates=[1, 3, 9]):
        super().__init__()
        
        # Initial convolution to process the latent representation
        self.conv_in = nn.Conv1d(
            latent_dim, channels[0],
            kernel_size=3,
            padding=1
        )
        self.norm_in = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU()
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_blocks.append(
                UpsamplingBlock(
                    channels[i], channels[i+1],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    dilation_rates=dilation_rates
                )
            )
        
        # Final projection to output channels (usually 1 for mono audio)
        self.proj = nn.Conv1d(channels[-1], out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.norm_in(self.conv_in(x)))
        
        # Upsampling blocks
        for block in self.up_blocks:
            x = block(x)
        
        # Project to output channels and apply tanh
        x = self.tanh(self.proj(x))
        
        return x