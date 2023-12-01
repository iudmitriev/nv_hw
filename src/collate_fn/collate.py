import logging
from typing import List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(dataset_items: List[dict]):
    
    result_batch = {}
    values_to_pad = ['spectrogram', 'audio']
    for value in values_to_pad:
        lengths = [item[value].shape[-1] for item in dataset_items]
        result_batch[f'{value}_length'] = torch.tensor(lengths)

        size_to_pad = max(lengths)
        result_batch[value] = torch.cat([
            F.pad(
                input = item[value], 
                pad = (0, size_to_pad - item[value].shape[-1]),
                value = 0
            )
            for item in dataset_items
        ])
        if value == 'audio':
            result_batch[value] = result_batch[value].unsqueeze(dim=1)
    return result_batch

