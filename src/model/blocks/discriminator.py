import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleScaleDisctiminator(nn.Module):
    def __init__(self, weight_normalizer=None):
        super().__init__()

        # All this weird constants are taken from appendix A of original MelGAN article
        convs = [
            nn.Conv1d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=15, 
                stride=1, 
                groups=1,
                padding='same'
            ),
            nn.Conv1d(
                in_channels=16, 
                out_channels=64, 
                kernel_size=41, 
                stride=4, 
                groups=4,
                padding=20
            ),
            nn.Conv1d(
                in_channels=64, 
                out_channels=256, 
                kernel_size=41, 
                stride=4, 
                groups=16,
                padding=20
            ),
            nn.Conv1d(
                in_channels=256, 
                out_channels=1024, 
                kernel_size=41, 
                stride=4, 
                groups=64,
                padding=20
            ),
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=41, 
                stride=4, 
                groups=256,
                padding=20
            ),
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=5, 
                stride=1, 
                groups=1,
                padding=20
            ),
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1, 
                kernel_size=3, 
                stride=1, 
                groups=1,
                padding='same'
            ),
        ]

        if weight_normalizer is not None:
            convs = [weight_normalizer(module) for module in convs]
        
        self.body = nn.ModuleList([])
        for i, conv in enumerate(convs):
            block = [conv]
            if i != len(convs) - 1:
                block.append(nn.LeakyReLU())

            self.body.append(nn.Sequential(*block))


    def forward(self, input):
        features = []
        for layer in self.body:
            input = layer(input)
            features.append(input)
        out = input.view(input.shape[0], -1)
        return out, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_discriminators = nn.ModuleList([
            SingleScaleDisctiminator(weight_normalizer=nn.utils.spectral_norm),
            SingleScaleDisctiminator(weight_normalizer=nn.utils.weight_norm),
            SingleScaleDisctiminator(weight_normalizer=nn.utils.weight_norm)
        ])

        self.avgpool1 = nn.AvgPool1d(
            kernel_size=4, 
            stride=2, 
            padding=2
        )
        self.avgpool2 = nn.AvgPool1d(
            kernel_size=4, 
            stride=2, 
            padding=2
        )

    def forward(self, input):
        all_features = []
        all_predictions = []
        for i, discriminator in enumerate(self.scale_discriminators):
            if i == 0:
                out = input
            elif i == 1:
                out = self.avgpool1(input)
            elif i == 2:
                out = self.avgpool2(self.avgpool1(input))
            out, features = discriminator(out)
            
            all_predictions.append(out)
            all_features.append(features)
        return {
            'predictions': all_predictions,
            'features': all_features
        }


class SinglePeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

        self.body = nn.ModuleList([])
        for i in range(5):
            in_channels = 2**(5 + i) if i != 0 else 1
            stride = (3, 1) if i != 4 else 1
            padding = 2 if i != 4 else 1
            self.body.append(
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=2**(6 + i),
                            kernel_size=(5, 1),
                            stride=stride,
                            padding=(padding, 0)
                        )
                    ),
                    nn.LeakyReLU()
                )
            )
        
        self.head = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1,
                kernel_size=(3, 1),
                stride=1,
                padding='same'
            )
        )
    
    def forward(self, input):
        if input.shape[-1] % self.period != 0:
            pad_amount = self.period - input.shape[-1] % self.period
            input = F.pad(
                input=input,
                pad=(0, pad_amount)
            ).to(input.device)
        out = input.view(input.shape[0], 1, -1, self.period)

        features = []
        for layer in self.body:
            out = layer(out)
            features.append(out)

        out = self.head(out)
        features.append(out)
        out = out.view(out.shape[0], -1)
        return out, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods):
        super().__init__()
        
        self.discriminators = nn.ModuleList([])
        for period in periods:
            self.discriminators.append(SinglePeriodDiscriminator(period=period))
    
    def forward(self, input):
        all_predictions = []
        all_features = []
        for discriminator in self.discriminators:
            out, features = discriminator(input)
            all_predictions.append(out)
            all_features.append(features)
        return {
            'predictions': all_predictions,
            'features': all_features
        }


class Discriminator(nn.Module):
    def __init__(self, periods):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator(periods)

    def forward(self, audio):
        mpd_output = self.mpd(audio)
        msd_output = self.msd(audio)
        output = {
            'mpd_predictions': mpd_output['predictions'],
            'msd_predictions': msd_output['predictions'],
            'mpd_features': mpd_output['features'],
            'msd_features': msd_output['features']
        }
        return output
