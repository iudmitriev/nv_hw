import torch
from src.base import BaseModel
from src.model.blocks import Discriminator, Generator
from torch import nn


class HiFiGANModel(nn.Module):
    def __init__(self, input_channels, hidden_dim, 
                 upsample_conv_kernel_sizes, upsample_conv_strides,
                 block_kernel_sizes, block_dilation_sizes, periods):
        super().__init__()
        
        self.generator = Generator(
            input_channels=input_channels, 
            hidden_dim=hidden_dim, 
            upsample_conv_kernel_sizes=upsample_conv_kernel_sizes, 
            upsample_conv_strides=upsample_conv_strides,
            block_kernel_sizes=block_kernel_sizes, 
            block_dilation_sizes=block_dilation_sizes
        )
        
        self.discriminator = Discriminator(
            periods=periods
        )

    def forward(self, spectrogram, **batch):
        return self.generator(spectrogram)

    def discriminate(self, audio, **batch):
        return self.discriminator(audio)
