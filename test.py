import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import hydra
import logging

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders

from src.utils.waveglow import get_WaveGlow
import waveglow as waveglow

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "checkpoints" / "checkpoint.pth"
DEFAULT_INPUT_PATH = ROOT_PATH / "test_texts.json"
DEFAULT_RESULTS_PATH = ROOT_PATH / "results"


@hydra.main(version_base=None, config_path="src", config_name="config")
def main(config):

    checkpoint_path, in_file, out_dir = parse_args()

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

    waveglow_object = get_WaveGlow().to(device)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    with open(in_file, 'r') as input_file:
        input_texts = json.load(input_file)
    
    out_dir = Path(out_dir).absolute().resolve()
    with torch.no_grad():
        for i, text in enumerate(input_texts):
            if isinstance(text, str):
                text = {'text': text}

            duration_control = text.get('duration_control', 1.0)
            pitch_control = text.get('pitch_control', 1.0)
            energy_control = text.get('energy_control', 1.0)
            text = text['text']

            src_sequence = torch.from_numpy(np.array(text_to_sequence(text, ["english_cleaners"])))

            src_positions = [np.arange(1, int(src_sequence.shape[0]) + 1)]
            src_positions = torch.from_numpy(np.array(src_positions)).to(device)
            src_sequence = src_sequence.unsqueeze(0).to(device)
            
            output = model(
                src_sequence=src_sequence, 
                src_positions=src_positions,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control
            )
            melspec = output["mel_prediction"].squeeze()
            mel = melspec.unsqueeze(0).contiguous().transpose(1, 2).to(device)

            file_name = f'test_{text[:10]}_{duration_control:.1f}_{pitch_control:.1f}_{energy_control:.1f}.wav'
            out_file = out_dir / file_name
            waveglow.inference.inference(
                mel, waveglow_object,
                str(out_file)
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
        "--texts",
        default=str(DEFAULT_INPUT_PATH),
        type=str,
        help="Path to json file, containing text pharases to turn into speech (default: test_texts.json)",
    )
    args = args.parse_args()
    return args.resume, args.texts, args.output


if __name__ == "__main__":
    main()
