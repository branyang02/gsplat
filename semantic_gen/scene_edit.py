from dataclasses import dataclass, field
from typing import List, Literal

import torch
import tyro

from semantic_gen.deformation_field.deformation import Deformation
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

    ### Deformation Field Parameters ###
    net_width: int = 128
    defor_depth: int = 0
    timebase_pe: int = 4
    posebase_pe: int = 10
    scale_rotation_pe: int = 2
    opacity_pe: int = 2
    timenet_width: int = 64
    timenet_output: int = 32
    grid_pe: int = 0

    # kplanes config
    grid_dimensions: int = 2
    input_coordinate_dim: int = 4
    output_coordinate_dim: int = 16
    resolution: List[int] = field(default_factory=lambda: [64, 64, 64, 150])

    no_grid: bool = False
    bounds: float = 1.6
    multires: List[int] = field(default_factory=lambda: [1, 2])
    empty_voxel: bool = False
    static_mlp: bool = False
    dx: bool = False
    ds: bool = False
    dr: bool = False
    do: bool = False
    dshs: bool = False
    apply_rotation: bool = False


class Runner:
    """Runner for scene editing"""

    def __init__(self, cfg: Config):
        set_random_seed(42)
        self.cfg = cfg
        self.device = cfg.device

        deformation = Deformation(cfg).to(self.device)
        print(deformation)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
    runner = Runner(cfg)
