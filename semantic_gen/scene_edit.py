from dataclasses import dataclass, field
import json
import math
import os
from typing import List, Literal

import torch
import tyro

from semantic_gen.deformation_field.deformation import Deformation
from semantic_gen.datasets.colmap import Parser, DynamicDataset
from semantic_gen.utils import SAMOptModule, set_random_seed


@dataclass
class Config:
    # checkpoint to the trained 3DGS scene
    ckpt: str
    # output path
    result_dir: str
    # viewer port
    port: int = 8080
    # device
    device: Literal["cpu", "cuda"] = "cuda"

    """temp parameters for goal scene"""
    goal_data_dir: str = "data/colmap"
    goal_data_factor: int = 4
    goal_test_every: int = 8

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

    def __init__(self, ckpt, cfg: Config):
        set_random_seed(42)

        self.cfg = cfg
        self.device = cfg.device

        # Output directories
        os.makedirs(cfg.result_dir, exist_ok=True)

        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # load deformation field
        self.deformation = Deformation(cfg).to(self.device)
        # load 3DGS scene from ckpt
        self.splats = ckpt["splats"]
        # load pretrained SAM module
        self.sam_module, self.sh_degree = self._get_sam_module(ckpt["sam_module"])

        """In this temporary implementation, we assume we have the gaol scene"""
        self.parser = Parser(
            data_dir=cfg.goal_data_dir,
            factor=cfg.goal_data_factor,
            normalize=True,
            test_every=cfg.goal_test_every,
        )
        self.trainset = DynamicDataset(os.path.join(cfg.goal_data_dir, "trainset"))
        self.valset = DynamicDataset(os.path.join(cfg.goal_data_dir, "valset"))

    # Copied from semantic_gen/sam_viewer.py
    def _get_sam_module(self, sam_state_dict):
        n = sam_state_dict["embeds.weight"].shape[0]
        feature_dim = sam_state_dict["feature_head.4.weight"].shape[0]
        embed_dim = sam_state_dict["embeds.weight"].shape[1]
        mlp_width = sam_state_dict["color_head.0.weight"].shape[0]
        sh_degree = int(
            math.sqrt(
                sam_state_dict["color_head.0.weight"].shape[1] - feature_dim - embed_dim
            )
            - 1
        )
        mlp_depth = len(sam_state_dict) // 2 - 4

        sam_module = SAMOptModule(
            n=n,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            sh_degree=sh_degree,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            output_dim=feature_dim,
        ).to(self.cfg.device)

        sam_module.load_state_dict(sam_state_dict)
        print("sam_module", sam_module)

        return sam_module, sh_degree

    def train(self):
        cfg = self.cfg

        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
    runner = Runner(ckpt, cfg)
