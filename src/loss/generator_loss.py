import torch
from torch import nn
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_features = 2, lambda_mel = 45, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_features = lambda_features
        self.lambda_mel = lambda_mel

    def forward(
        self,
        mel_prediction,
        mel_target,
        discriminator_predictions=None,
        feature_predictions=None,
        feature_target=None,
        **kwargs
    ):
        loss = {}

        loss['mel_loss'] = F.l1_loss(mel_prediction, mel_target)

        if discriminator_predictions is not None:
            loss['generator_advantage'] = 0
            for family_discriminator_prediction in discriminator_predictions:
                for discriminator_prediction in family_discriminator_prediction:
                    loss['generator_advantage'] += torch.mean((1 - discriminator_prediction)**2)

        if feature_predictions is not None and feature_target is not None:
            feature_loss = 0
            for generated_features, real_features in zip(feature_predictions, feature_target):
                for real_feature, denerated_feature in zip(generated_features, real_features):
                    feature_loss += torch.mean(torch.abs(real_feature - denerated_feature))
            loss['feature_loss'] = feature_loss

        if 'generator_advantage' in loss and 'feature_loss' in loss:
            loss['generator_loss'] = loss['generator_advantage'] + \
                                     loss['feature_loss'] * self.lambda_features + \
                                     loss['mel_loss'] * self.lambda_mel
        
        return loss
