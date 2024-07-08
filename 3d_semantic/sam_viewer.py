from dataclasses import dataclass
from fast_pytorch_kmeans import KMeans
import math
import time
from typing import Dict, Literal, Tuple

import tyro

from examples.datasets.clip import OpenCLIPNetwork, OpenCLIPNetworkConfig
import nerfview
import torch
from torch import Tensor
import viser
from examples.utils import SAMOptModule
from gsplat.rendering import rasterization

from gsplat._helper import *


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

        # clip
        self.clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        print("clip_model", self.clip)

        # others
        self.feature_colors = None

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

        ## TODO: create color clone that only displays feature colors
        if kwargs.get("segment", False):
            if self.feature_colors is None:
                # run K means on features
                self.feature_colors = colors.clone()
                num_clusters = 20
                kmeans = KMeans(
                    n_clusters=num_clusters,
                    mode="cosine",
                    verbose=1,
                    max_iter=1000,
                )
                labels = kmeans.fit_predict(self.feature_colors.squeeze(0))
                palette = (
                    torch.randint(0, 256, (num_clusters, 3)).float().cuda() / 255.0
                )
                for cluster in range(num_clusters):
                    cluster_indices = (labels == cluster).nonzero(as_tuple=True)[0]
                    self.feature_colors[0, cluster_indices] = palette[cluster]

            colors = self.feature_colors

        features = features + self.splats["features"]
        colors_with_features = torch.cat([colors, features], dim=-1)

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
        print("feature_query: ", feature_query)

        tok_phrase = self.clip.tokenizer(feature_query).to(self.cfg.device)
        feature_embeds = self.clip.model.encode_text(tok_phrase)  # [1, 512]
        feature_embeds /= feature_embeds.norm(dim=-1, keepdim=True)

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
            feature_embeds=feature_embeds,
            # segment=True,
            **kwargs,
        )
        colors = render_colors[..., :3]  # [1, H, W, 3]
        render_features = render_colors[..., 3:]  # [1, H, W, 512]

        feature_embeds = feature_embeds.view(1, 1, 1, 512)
        cosine_similarity = F.cosine_similarity(render_features, feature_embeds, dim=-1)

        threshold = kwargs.get("feature_similarity_threshold", 0.3)

        mask = cosine_similarity > threshold

        new_color = torch.tensor([1.0, 0.0, 0.0], device=colors.device)
        colors[mask] = new_color

        return colors[0].cpu().numpy()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    torch.manual_seed(42)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
    renderer = Renderer(ckpt, cfg)
    renderer.start_server()
