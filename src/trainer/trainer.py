import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker

import torchaudio


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            generator_loss,
            discriminator_loss,
            generator_optimizer,
            discriminator_optimizer,
            config,
            device,
            dataloaders,
            spectrogram,
            generator_lr_scheduler=None,
            discriminator_lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            test_wavs=None
    ):
        super().__init__(
            model=model, 
            generator_optimizer = generator_optimizer, 
            discriminator_optimizer = discriminator_optimizer, 
            generator_lr_scheduler = generator_lr_scheduler, 
            discriminator_lr_scheduler = discriminator_lr_scheduler,
            config = config,
            device = device
        )

        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        self.spectrogram = spectrogram.to(device)

        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "generator_loss",
            "generator_advantage",
            "feature_loss",
            "mel_loss",
            "discriminator_loss",
            "msd_discriminator_loss",
            "msd_discriminator_generated_loss", 
            "msd_discriminator_real_loss",
            "mpd_discriminator_loss",
            "mpd_discriminator_generated_loss", 
            "mpd_discriminator_real_loss",
            "grad norm",
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "mel_loss", writer=self.writer
        )

        self.test_wavs = test_wavs


    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        tensors = [
            "spectrogram",
            "audio"
        ]
        for tensor_for_gpu in tensors:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self._log_audio(batch['predicted_audio'][0, :, :], name='predicted_audio')
            self._log_audio(batch['audio'][0, :, :], name='real_audio')
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "Generator learning rate", self.generator_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "Discriminator learning rate", self.discriminator_lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()
        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()
        
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["predicted_audio"] = outputs

        if not is_train:
            mel_prediction = self.spectrogram(batch["predicted_audio"])
            real_audio = F.pad(
                input=batch['audio'], 
                pad=(0, batch["predicted_audio"].shape[-1] - batch['audio'].shape[-1])
            )
            mel_target = self.spectrogram(real_audio)
            generator_losses = self.generator_loss(
                mel_prediction=mel_prediction,
                mel_target=mel_target
            )
            batch.update(generator_losses)

            metrics.update('mel_loss', batch['mel_loss'].item())
        else:
            discriminator_generated_pred = self.model.discriminate(batch["predicted_audio"].detach())
            discriminator_real_pred = self.model.discriminate(batch["audio"])
            
            msd_discriminator_losses = self.discriminator_loss(
                generated_predictions = discriminator_generated_pred['msd_predictions'], 
                real_predictions = discriminator_real_pred['msd_predictions']
            )
            mpd_discriminator_losses = self.discriminator_loss(
                generated_predictions = discriminator_generated_pred['mpd_predictions'], 
                real_predictions = discriminator_real_pred['mpd_predictions']
            )
            discriminator_loss = (msd_discriminator_losses['discriminator_loss'] + 
                                  mpd_discriminator_losses['discriminator_loss'])
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            batch.update(msd_discriminator_losses)
            batch.update(mpd_discriminator_losses)
            metrics.update('discriminator_loss', discriminator_loss.item())
            for loss_name in ["discriminator_loss",
                              "discriminator_generated_loss", 
                              "discriminator_real_loss",
                             ]:
                metrics.update(f'mpd_{loss_name}', mpd_discriminator_losses[loss_name].item())
                metrics.update(f'msd_{loss_name}', msd_discriminator_losses[loss_name].item())

            batch['real_audio'] = F.pad(
                input=batch['audio'], 
                pad=(0, batch["predicted_audio"].shape[-1] - batch['audio'].shape[-1])
            )
            mel_prediction = self.spectrogram(batch["predicted_audio"])
            mel_target = self.spectrogram(batch['real_audio'])

            discriminator_generated = self.model.discriminate(batch["predicted_audio"])
            discriminator_real = self.model.discriminate(batch["real_audio"])

            discriminator_predictions = [discriminator_generated['mpd_predictions'], 
                                         discriminator_generated['msd_predictions']]
            feature_predictions = (discriminator_generated['mpd_features'] + 
                                   discriminator_generated['msd_features'])
            feature_target = (discriminator_real['mpd_features'] + 
                              discriminator_real['msd_features'])

            generator_losses = self.generator_loss(
                mel_prediction=mel_prediction,
                mel_target=mel_target,
                discriminator_predictions=discriminator_predictions,
                feature_predictions=feature_predictions,
                feature_target=feature_target,
            )

            generator_losses['generator_loss'].backward()
            self.generator_optimizer.step()

            batch.update(generator_losses)
            for loss_name in ["generator_loss",
                              "generator_advantage",
                              "feature_loss",
                              "mel_loss"]:
                metrics.update(loss_name, batch[loss_name].item())
            
            self._clip_grad_norm()

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self._log_audio(batch["predicted_audio"][0, :, :], name='predicted_audio')
            self._log_audio(batch["audio"][0, :, :], name='real_audio')
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            if self.test_wavs is not None:
                self._log_test_audio()
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_audio(self, audio, name):
        audio = audio.cpu()
        self.writer.add_audio(name, audio, sample_rate=self.config["preprocessing"]["sr"])


    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_test_audio(self):
        for i, audio in enumerate(self.test_wavs):
            self.model.eval()
            audio = audio.to(self.device)
            spectrogram = self.spectrogram(audio)
            predicted_audio = self.model(spectrogram)["predicted_audio"].cpu()
            predicted_audio = predicted_audio.squeeze(dim=0)
            audio = audio.cpu()
            self.writer.add_audio(f"test_real_audio_{i}", audio, sample_rate=self.config["preprocessing"]["sr"])
            self.writer.add_audio(f"test_predicted_audio_{i}", predicted_audio, sample_rate=self.config["preprocessing"]["sr"])
        