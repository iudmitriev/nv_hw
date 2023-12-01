import torch
from torch import nn
import torch.nn.functional as F

class DiscriminatorLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        generated_predictions,
        real_predictions,
        **kwargs
    ):
        generated_loss = 0
        real_loss = 0
        for prediction, real in zip(generated_predictions,
                                    real_predictions):
            generated_loss += torch.mean(prediction**2)
            real_loss += torch.mean((1 - real)**2)
        loss = generated_loss + real_loss
        return {
            'discriminator_loss': loss, 
            'discriminator_generated_loss': generated_loss, 
            'discriminator_real_loss': real_loss
        }
