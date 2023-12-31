from operator import xor

import hydra

from torch.utils.data import ConcatDataset, DataLoader
from omegaconf.dictconfig import DictConfig

import src.augmentations
import src.datasets
from src.collate_fn.collate import collate_fn


def get_dataloaders(config: DictConfig):
    dataloaders = {}
    for split, params in config["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = src.augmentations.from_configs(config)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds_name, ds in params["datasets"].items():
            datasets.append(
                hydra.utils.instantiate(
                    ds,
                    main_config=config,  
                    wave_augs=wave_augs, 
                    spec_augs=spec_augs,
                    _recursive_=False
                )
            )
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = hydra.utils.instantiate(params["batch_sampler"], data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
