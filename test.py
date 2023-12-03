import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import hydra
import logging

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "checkpoints" / "checkpoint.pth"
DEFAULT_INPUT_PATH = ROOT_PATH / "test_wavs"
DEFAULT_RESULTS_PATH = ROOT_PATH / "results"


@hydra.main(version_base=None, config_path="src", config_name="config")
def main(config):

    checkpoint_path, in_dir, out_dir = parse_args()

    logger = logging.getLogger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = hydra.utils.instantiate(config["arch"])
    logger.info(model)


    logger.info(f"Loading checkpoint: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    spectrogram_generator = hydra.utils.instantiate(config["preprocessing"]["spectrogram"])
    spectrogram_generator = spectrogram_generator.to(device)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    in_dir = Path(in_dir)
    test_wavs = {}
    for wav in in_dir.iterdir():
        audio_tensor, sr = torchaudio.load(wav)
        assert sr == config["preprocessing"]["sr"]
        test_wavs[wav] = audio_tensor
    
    
    for wav, audio in test_wavs.items():
        audio = audio.to(device)
        spectrogram = spectrogram_generator(audio)
        predicted_audio = model(spectrogram)["predicted_audio"].cpu()
        predicted_audio = predicted_audio.squeeze(dim=0)
        out_file = Path(out_dir) / wav.name
        torchaudio.save(
            uri=out_file,
            src=predicted_audio, 
            sample_rate=config["preprocessing"]["sr"]
        )


def parse_args():
    args = argparse.ArgumentParser(description="Pytorch model test")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH),
        type=str,
        help="path to latest checkpoint (default: checkpoints/checkpoint.pth)",
    )
    args.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        type=str,
        help="Folder to write results (default: results/)",
    )
    args.add_argument(
        "-t",
        "--test",
        default=str(DEFAULT_INPUT_PATH),
        type=str,
        help="Path to directory, containing audio to test (default: test_wavs/)",
    )
    args = args.parse_args()
    return args.resume, args.test, args.output


if __name__ == "__main__":
    main()
