import argparse
import math
import time
from typing import Dict, Tuple

import nerfview
import torch
from torch import Tensor
import viser
from examples.utils import SAMOptModule
from gsplat.rendering import rasterization

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="results/", help="where to dump outputs"
)
parser.add_argument("--ckpt", type=str, default=None, help="path to the .pt file")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument(
    "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
)
args = parser.parse_args()

torch.manual_seed(42)
device = "cuda"


# global sh_degree
sh_degree = None
ckpt = torch.load(args.ckpt, map_location=device)
splats = ckpt["splats"]


def _get_sam_module(sam_state_dict):
    global sh_degree

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
    ).to(device)

    sam_module.load_state_dict(sam_state_dict)
    print("sam_module", sam_module)

    return sam_module


def rasterize_splats(
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    splats: Dict[str, Tensor] = splats,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    global sh_degree

    means = splats["means3d"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])

    image_ids = kwargs.pop("image_ids", None)

    # get colors from sam_module
    colors, features = sam_module(
        features=splats["features"],
        embed_ids=image_ids,
        dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        sh_degree=sh_degree,
    )
    colors = colors + splats["colors"]
    colors = torch.sigmoid(colors)
    features = features + splats["features"]
    colors_with_features = torch.cat([colors, features], dim=-1)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
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


# register and open viewer
@torch.no_grad()
def viewer_render_fn(
    camera_state: nerfview.CameraState, img_wh: Tuple[int, int], **kwargs
):
    feature_query = kwargs.get("feature_query", None)
    print("feature_query", feature_query)

    W, H = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)

    render_colors, _, _ = rasterize_splats(
        camtoworlds=c2w[None],
        Ks=K[None],
        width=W,
        height=H,
        radius_clip=3.0,  # skip GSs that have small image radius (in gt_colors)
    )  # [1, H, W, 3]
    print(render_colors.shape)
    return render_colors[0].cpu().numpy()


if __name__ == "__main__":

    ## other things
    sam_state_dict = ckpt["sam_module"]
    sam_module = _get_sam_module(sam_state_dict)

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)
