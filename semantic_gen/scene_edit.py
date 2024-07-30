from dataclasses import dataclass, field
import json
import math
import os
import shutil
import time
from typing import Dict, List, Literal, Tuple

import imageio
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision

from datasets.traj import generate_interpolated_path

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tqdm
import tyro
import viser
from gsplat.rendering import rasterization
import nerfview

from semantic_gen.deformation_field.deformation import Deformation
from semantic_gen.datasets.colmap import Parser, DynamicDataset
from semantic_gen.utils import SAMOptModule, set_random_seed, normalized_quat_to_rotmat


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [batch, c, h-1, w]
    second_difference = (
        first_difference[..., 1:, :] - first_difference[..., : h - 2, :]
    )  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


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

    """params copied from semantic_gen/sam_trainer.py"""
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Learning rate for SAM optimization
    sam_opt_lr: float = 1e-3
    # Regularization for SAM optimization as weight decay
    sam_opt_reg: float = 1e-6
    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # Disable viewer
    disable_viewer: bool = False
    # Number of training steps
    max_steps: int = 30_000
    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Language features lambda
    language_features_lambda: float = 0.2
    # Weight for depth loss
    depth_lambda: float = 1e-2
    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1
    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [3_000, 7_000, 14_000, 30_000]
    )
    # Steps to save the model
    save_steps: List[int] = field(
        default_factory=lambda: [3_000, 7_000, 14_000, 30_000]
    )

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
    no_dx: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = False
    no_dshs: bool = False
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

        # Tensorboard
        tb_log_dir = f"{cfg.result_dir}/tb"
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # load deformation field
        self.deformation = Deformation(cfg).to(self.device)
        print(self.deformation)
        # load 3DGS scene from ckpt
        self.splats = torch.nn.ParameterDict(ckpt["splats"])
        for key, value in self.splats.items():
            self.splats[key].requires_grad = True
        # load pretrained SAM module
        self.sam_module, self.sh_degree = self._get_sam_module(ckpt["sam_module"])

        xyz_max = self.splats["means3d"].max(dim=0).values.detach().cpu().numpy()
        xyz_min = self.splats["means3d"].min(dim=0).values.detach().cpu().numpy()
        self.deformation.deformation_net.set_aabb(xyz_min, xyz_max)

        """In this temporary implementation, we assume we have the gaol scene"""
        self.parser = Parser(
            data_dir=cfg.goal_data_dir,
            factor=cfg.goal_data_factor,
            normalize=True,
            test_every=cfg.goal_test_every,
        )
        self.trainset = DynamicDataset(os.path.join(cfg.goal_data_dir, "trainset"))
        self.valset = DynamicDataset(os.path.join(cfg.goal_data_dir, "valset"))
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.feature_dim = self.trainset.feature_dim
        print("Feature dim:", self.feature_dim)

        self.optimizers = self._create_optimizers_from_splats()
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        # Optimizers for Deformation Field
        self.deformation_optimizers = [
            torch.optim.Adam(
                # deformation
                self.deformation.get_mlp_parameters(),
                lr=1e-3 * math.sqrt(cfg.batch_size),
                weight_decay=1e-6,
            ),
            torch.optim.Adam(
                # grid
                self.deformation.get_grid_parameters(),
                lr=1e-3 * math.sqrt(cfg.batch_size),
                weight_decay=1e-6,
            ),
        ]

        # Modified from semantic_gen/sam_trainer.py
        self.app_optimizers = [
            torch.optim.Adam(
                self.sam_module.embeds.parameters(),
                lr=cfg.sam_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                weight_decay=cfg.sam_opt_reg,
            ),
            torch.optim.Adam(
                self.sam_module.color_head.parameters(),
                lr=cfg.sam_opt_lr * math.sqrt(cfg.batch_size),
            ),
            torch.optim.Adam(
                self.sam_module.feature_head.parameters(),
                lr=cfg.sam_opt_lr * math.sqrt(cfg.batch_size),
            ),  ##### added optimizer for feature_head
        ]

        # Losses & Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        self.mse = torch.nn.MSELoss().to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.server.request_share_url()
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    # Modified from semantic_gen/sam_trainer.py
    def train(self):
        cfg = self.cfg
        device = self.device

        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        scheulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            for key in data:
                data[key] = data[key].squeeze(0)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            gt_colors = data["image"][..., :3].to(device) / 255.0  # [1, H, W, 3]
            gt_language_features = data["image"][..., 3:].to(device)
            gt_language_features_mask = data["point_feature"].to(device)
            num_train_rays_per_step = (
                gt_colors.shape[0] * gt_colors.shape[1] * gt_colors.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if "depths" in data:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
                cfg.depth_loss = True

            height, width = gt_colors.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, self.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )

            if renders.shape[-1] == 3 + self.feature_dim:
                colors, features = renders[..., :3], renders[..., 3:]
            elif renders.shape[-1] == 3 + self.feature_dim + 1:
                assert cfg.depth_loss == True
                colors, features = (
                    renders[..., :3],
                    renders[..., 3 : 3 + self.feature_dim],
                )
                depths = renders[..., 3 + self.feature_dim :]
            else:
                raise ValueError(f"Invalid number of channels: {renders.shape[-1]}")

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # Debugging
            # torchvision.utils.save_image(
            #     colors[0].permute(2, 0, 1),
            #     f"{cfg.result_dir}/renders/{step:06d}_colors.png",
            # )

            info["means2d"].retain_grad()  # used for running stats

            # loss
            ##### TODO: verify the correctness of the loss function for gt_language_features.
            l1loss_colors = F.l1_loss(colors, gt_colors)  ##### changed variable name
            l1loss_features = F.l1_loss(
                features * gt_language_features_mask.unsqueeze(-1),
                gt_language_features * gt_language_features_mask.unsqueeze(-1),
            )
            ssimloss = 1.0 - self.ssim(
                gt_colors.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = (
                l1loss_colors * (1.0 - cfg.ssim_lambda)
                + ssimloss * cfg.ssim_lambda
                + l1loss_features * cfg.language_features_lambda
            )  ##### added l1loss_features
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            # Plane Loss
            tv_loss = self.compute_regulation()
            loss += tv_loss

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar(
                    "train/l1loss_colors", l1loss_colors.item(), step
                )  ##### changed variable name
                self.writer.add_scalar(
                    "train/l1loss_features", l1loss_features.item(), step
                )  ##### added l1loss_features
                self.writer.add_scalar("train/tv_loss", tv_loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([gt_colors, colors], dim=2).detach().cpu().numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(info)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()

                if step % cfg.reset_every == 0:
                    self.reset_opa(cfg.prune_opa * 2.0)

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.deformation_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in scheulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                        "sam_module": self.sam_module.state_dict(),  ##### Save state of self.sam_module
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    # Copied from semantic_gen/sam_trainer.py
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        colors, features = self.sam_module(
            features=self.splats["features"],
            embed_ids=image_ids,
            dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
            sh_degree=kwargs.pop("sh_degree", self.sh_degree),
        )
        colors = colors + self.splats["colors"]
        colors = torch.sigmoid(colors)
        features = features + self.splats["features"]
        colors_with_features = torch.cat(
            [colors, features], -1
        )  # [(C,), N, 3 + feature_dim]

        # Deformation field
        shs = torch.randn((means.shape[0], 16, 3), device=self.device)
        time = torch.randn((means.shape[0], 1), device=self.device)

        means3D_final, scales_final, quats_final, opacity_final, shs_final = (
            self.deformation(means, scales, quats, opacities.unsqueeze(-1), shs, time)
        )

        # Copied from 4DGS
        scales = torch.exp(scales_final)
        quats = F.normalize(quats_final)
        opacities = torch.sigmoid(opacity_final).squeeze(-1)
        means = means3D_final

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_with_features,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info

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

    # Modified from semantic_gen/sam_trainer.py
    def _create_optimizers_from_splats(self, batch_size=1):
        params = [
            # name, value, lr
            ("means3d", self.splats["means3d"], 1.6e-4 * self.scene_scale),
            ("scales", self.splats["scales"], 5e-3),
            ("quats", self.splats["quats"], 1e-3),
            ("opacities", self.splats["opacities"], 5e-2),
            ("features", self.splats["features"], 2.5e-3),
            ("colors", self.splats["colors"], 2.5e-3),
        ]
        optimizers = [
            torch.optim.Adam(
                [
                    {
                        "params": self.splats[name],
                        "lr": lr * math.sqrt(batch_size),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        ]

        return optimizers

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size
        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids))
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "mse": []}  ##### added mse
        for i, data in enumerate(valloader):
            for key in data:
                data[key] = data[key].squeeze(0)  ##### add 1 extra dimension

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            gt_colors = data["image"][..., :3].to(device) / 255.0
            gt_language_features = data["image"][..., 3:].to(device)
            height, width = gt_colors.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            if (
                renders.shape[-1] == 3 + self.feature_dim
            ):  ##### evaluate featues as well
                colors, features = renders[..., :3], renders[..., 3:]
            else:
                colors = renders
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([gt_colors, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            gt_colors = gt_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, gt_colors))
            metrics["ssim"].append(self.ssim(colors, gt_colors))
            metrics["lpips"].append(self.lpips(colors, gt_colors))
            metrics["mse"].append(
                self.mse(features, gt_language_features)
            )  ##### added mse

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        mse = torch.stack(metrics["mse"]).mean()  ##### added mse
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f}, MSE: {mse.item():.3f}\n"
            f"Time: {ellipse_time:.3f}s/image \n"
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "mse": mse.item(),  ##### added mse
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=self.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int], **kwargs
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in gt_colors)
            **kwargs,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    def _plane_regulation(self):
        multi_res_grids = self.deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self.deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self.deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(
        self,
        time_smoothness_weight=0.001,
        l1_time_planes_weight=0.001,
        plane_tv_weight=0.002,
    ):
        return (
            plane_tv_weight * self._plane_regulation()
            + time_smoothness_weight * self._time_regulation()
            + l1_time_planes_weight * self._l1_regulation()
        )


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
    runner = Runner(ckpt, cfg)
    runner.train()
