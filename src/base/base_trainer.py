from abc import abstractmethod

import torch
from numpy import inf
import logging
from pathlib import Path

from src.base import BaseModel
from src.logger import get_visualizer
from src.utils import get_savedir


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
            self, model: BaseModel, 
            generator_optimizer, 
            discriminator_optimizer, 
            generator_lr_scheduler, 
            discriminator_lr_scheduler, 
            config, 
            device
        ):
        self.device = device
        self.config = config

        self.logger = logging.getLogger("train")
        self.logger.setLevel(logging.DEBUG)

        self.model = model
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = get_savedir(config)

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if "resume" in config:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "generator_lr_scheduler": self.generator_lr_scheduler.state_dict(),
            "discriminator_lr_scheduler": self.discriminator_lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        checkpoint_config = checkpoint["config"]
        if (
                (checkpoint_config["optimizer"]["generator_optimizer"] != 
                 self.config["optimizer"]["generator_optimizer"]) or
                (checkpoint_config["optimizer"]["discriminator_optimizer"] != 
                 self.config["optimizer"]["discriminator_optimizer"])
        ):
            self.logger.warning(
                "Warning: Optimizer given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])        

        if (
                (checkpoint_config["lr_scheduler"]["generator_lr_scheduler"] != 
                 self.config["lr_scheduler"]["generator_lr_scheduler"]) or
                (checkpoint_config["lr_scheduler"]["discriminator_lr_scheduler"] != 
                 self.config["lr_scheduler"]["discriminator_lr_scheduler"])
        ):
            self.logger.warning(
                "Warning: lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.generator_lr_scheduler.load_state_dict(checkpoint["generator_lr_scheduler"])
            self.discriminator_lr_scheduler.load_state_dict(checkpoint["discriminator_lr_scheduler"])


        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
