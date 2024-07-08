import os
from dataclasses import dataclass
from typing import Literal

import torch
import tqdm
import tyro
from datasets.colmap import Dataset, Parser
import torch.multiprocessing as mp


@dataclass
class Config:
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    test_every: int = 8
    depth_loss: bool = False
    sam_ckpt: str = "ckpts/sam_vit_h_4b8939.pth"
    num_workers: int = 4
    feature_level: Literal["default", "large", "medium", "small", "l", "m", "s"] = (
        "default"
    )


class PreProcessor:
    def __init__(self, cfg: Config):
        train_dir = os.path.join(cfg.data_dir, "trainset")
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(cfg.data_dir, "valset")
        os.makedirs(val_dir, exist_ok=True)

        if cfg.num_workers > 0:
            mp.set_start_method("spawn")

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            load_depths=cfg.depth_loss,
            sam_ckpt=cfg.sam_ckpt,
            feature_level=cfg.feature_level,
        )
        self.valset = Dataset(
            self.parser,
            split="val",
            sam_ckpt=cfg.sam_ckpt,
            feature_level=cfg.feature_level,
        )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        for i, batch in enumerate(tqdm.tqdm(trainloader)):
            torch.save(batch, f"{train_dir}/{i}.pt")

        # Explicitly delete trainloader and batches
        del trainloader
        del self.trainset
        torch.cuda.empty_cache() 

        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        for i, batch in enumerate(tqdm.tqdm(valloader)):
            torch.save(batch, f"{val_dir}/{i}.pt")


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    preprocessor = PreProcessor(cfg)
    print("Preprocessing done.")
