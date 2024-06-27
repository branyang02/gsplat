from dataclasses import dataclass
import math
import time
from typing import Dict, Literal, Tuple

import tyro

import nerfview
import torch
from torch import Tensor
import viser
from examples.utils import SAMOptModule
from gsplat.rendering import rasterization


@dataclass
class Config:
    ckpt: str
    output_dir: str = "results/"
    port: int = 8080
    backend: Literal["gsplat", "gsplat_legacy", "inria"] = "gsplat"
    device: Literal["cpu", "cuda"] = "cuda"


class Renderer:

    def __init__(
        self,
        ckpt: str,
        cfg: Config,
    ):
        self.cfg = cfg
        self.sam_module, self.sh_degree = self._get_sam_module(
            ckpt["sam_module"]
        )  # TODO: check sh_degree logic
        self.splats = ckpt["splats"]

    def start_server(self):

        self.server = viser.ViserServer(port=self.cfg.port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.viewer_render_fn,
            mode="rendering",
        )
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(100000)

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
        mlp_depth = len(sam_state_dict) // 2 - 4  ### TODO: double check this

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

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        means = self.splats["means3d"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)

        # get colors from sam_module
        colors, features = self.sam_module(
            features=self.splats["features"],
            embed_ids=image_ids,
            dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
            sh_degree=self.sh_degree,
        )
        colors = colors + self.splats["colors"]
        colors = torch.sigmoid(colors)
        features = features + self.splats["features"]
        colors_with_features = torch.cat([colors, features], dim=-1)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,  ## TODO: render color vs colors_with_features
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            sh_degree=None,
            **kwargs,
        )
        return render_colors, render_alphas, info

    @torch.no_grad()
    def viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int], **kwargs
    ):
        feature_query = kwargs.get("feature_query", None)
        print("feature_query", feature_query)

        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.cfg.device)
        K = torch.from_numpy(K).float().to(self.cfg.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            radius_clip=3.0,  # skip GSs that have small image radius (in gt_colors)
        )  # [1, H, W, 3]
        print(render_colors.shape)
        return render_colors[0].cpu().numpy()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    torch.manual_seed(42)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
    renderer = Renderer(ckpt, cfg)
    renderer.start_server()
