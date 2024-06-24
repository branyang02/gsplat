import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

from gsplat.rendering import rasterization


@dataclass
class Config:
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    test_every: int = 8
    patch_size: Optional[int] = None
    depth_loss: bool = False


class PreProcessor:
    def __init__(self, cfg: Config):

        train_dir = f"{cfg.data_dir}/trainset"
        os.makedirs(train_dir, exist_ok=True)
        val_dir = f"{cfg.data_dir}/valset"
        os.makedirs(val_dir, exist_ok=True)

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )

        for i, batch in enumerate(tqdm.tqdm(trainloader)):
            torch.save(batch, f"{train_dir}/{i}.pt")

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        for i, batch in enumerate(tqdm.tqdm(valloader)):
            torch.save(batch, f"{val_dir}/{i}.pt")


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    preprocessor = PreProcessor(cfg)
    print("Preprocessing done.")
