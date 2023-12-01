import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, input_channels, kernel_size, layer_dilation_sizes):
        super().__init__()
        self.layers = nn.ModuleList([])
        for dilation_sizes in layer_dilation_sizes:
            layer = []
            for dilation_size in dilation_sizes:
                layer.append(nn.LeakyReLU())
                layer.append(
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=input_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding='same'
                    )
                )
            layer = nn.Sequential(*layer)
            self.layers.append(layer)
    
    def forward(self, input):
        result = 0
        for layer in self.layers:
            result = result + layer(input)
        return result


class MRF(nn.Module):
    def __init__(self, input_channels, block_kernel_sizes, block_dilation_sizes):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for kernel_size, layer_dilation_sizes in zip(block_kernel_sizes, block_dilation_sizes):
            self.layers.append(
                ResBlock(
                    input_channels = input_channels,
                    kernel_size=kernel_size,
                    layer_dilation_sizes=layer_dilation_sizes
                )
            )
    
    def forward(self, input):
        out = 0
        for block in self.layers:
            out += block(input)
        return out


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_dim, 
                 upsample_conv_kernel_sizes, upsample_conv_strides,
                 block_kernel_sizes, block_dilation_sizes):
        super().__init__()

        self.tail = nn.Conv1d(
            in_channels = input_channels, 
            out_channels = hidden_dim, 
            kernel_size = 7,
            stride = 1,
            dilation = 1,
            padding = 3
        )

        body = []
        current_channels = hidden_dim
        for kernel_size, stride in zip(upsample_conv_kernel_sizes, upsample_conv_strides):
            upsample_conv = nn.ConvTranspose1d(
                in_channels=current_channels,
                out_channels=current_channels // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2
            )
            mrf_block = MRF(
                input_channels=current_channels // 2,
                block_kernel_sizes=block_kernel_sizes,
                block_dilation_sizes=block_dilation_sizes
            )
            body.append(upsample_conv)
            body.append(mrf_block)
            current_channels //= 2
        self.body = nn.Sequential(*body)

        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=current_channels, 
                out_channels=1, 
                kernel_size=7,
                stride=1,
                dilation=1, 
                padding='same'
            ),
            nn.Tanh()
        )
    
    def forward(self, spectrogram):
        out = self.tail(spectrogram)
        out = self.body(out)
        out = self.head(out)
        out = {
            'predicted_audio': out
        }
        return out
