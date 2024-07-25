from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from utils import set_random_seed


@dataclass
class Config:
    # checkpoint to the trained 3DGS scene
    ckpt: str
    # output path
    output: str = "results"
    # viewer port
    port: int = 8080
    # device
    device: Literal["cpu", "cuda"] = "cuda"


class Runner:
    """Runner for scene editing"""

    def __init__(self, cfg: Config):
        set_random_seed(42)
        self.cfg = cfg
        self.device = cfg.device

    # TODO:
    """
    1. load existing 3dGS
    2. perform edits giving new images
    3. yurr
    """


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
