import math
from typing import Dict, Tuple


from semantic_gen.datasets.clip import OpenCLIPNetwork, OpenCLIPNetworkConfig
import nerfview
import torch
from torch import Tensor
from semantic_gen.utils import SAMOptModule
from gsplat.rendering import rasterization

from gsplat._helper import *

import matplotlib.pyplot as plt


import torch

ckpt = torch.load(
    "../semantic_gen/results/nerf_dff_dataset_depth/ckpts/ckpt_29999.pt", map_location="cuda"
)


def _get_sam_module(sam_state_dict):
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
    ).to("cuda")

    sam_module.load_state_dict(sam_state_dict)
    print("sam_module", sam_module)

    return sam_module, sh_degree


sam_module, sh_degree = _get_sam_module(ckpt["sam_module"])

splats = ckpt["splats"]

clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)

import pickle

with open("viewer_render_fn_inputs.pkl", "rb") as f:
    inputs = pickle.load(f)


@torch.no_grad()
def viewer_render_fn(
    camera_state: nerfview.CameraState, img_wh: Tuple[int, int], **kwargs
):
    feature_query = kwargs.get("feature_query", None)
    print("feature_query: ", feature_query)

    tok_phrase = clip.tokenizer(feature_query).to("cuda")
    feature_embeds = clip.model.encode_text(tok_phrase)  # [1, 512]
    feature_embeds /= feature_embeds.norm(dim=-1, keepdim=True)

    W, H = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to("cuda")
    K = torch.from_numpy(K).float().to("cuda")

    render_colors, _, info, features = rasterize_splats(
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

    if kwargs.get("segment", False):
        render_features = render_colors[..., 3:]  # [1, H, W, 512]

        feature_embeds = feature_embeds.view(1, 1, 1, 512)
        cosine_similarity = F.cosine_similarity(render_features, feature_embeds, dim=-1)

        threshold = kwargs.get("feature_similarity_threshold", 0.3)

        mask = cosine_similarity > threshold  # torch.Size([1, 928, 2048])

        new_color = torch.tensor([1.0, 0.0, 0.0], device=colors.device)

        colors[mask] = new_color

    meta = {
        "color_torch": render_colors[..., :3],
        "segmentation_torch": colors,
        "mask_torch": mask if kwargs.get("segment", False) else None,
    }

    return colors[0].cpu().numpy(), features, info, meta


def rasterize_splats(
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    means = splats["means3d"]  # torch.Size([386490, 3])
    print("NUM OF GAUSSIANS: ", means.shape[0])
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
    colors = torch.sigmoid(colors)  # torch.Size([1, 386490, 3])

    features = features + splats["features"]
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

    return render_colors, render_alphas, info, features


query = "banana"
render_image, features, info, meta = viewer_render_fn(
    camera_state=inputs["camera_state"],
    img_wh=inputs["img_wh"],  # (2048, 928)
    feature_query=query,
    feature_similarity_threshold=0.28,
    compute_mapping=True,
    segment=True,
)


plt.imsave("images/render_image.png", render_image)

### move GSs
print("mapping size: ", info["mapping"].shape)
print("mask size: ", meta["mask_torch"].shape)

mapping = info["mapping"]
mask = meta["mask_torch"]

expanded_mask = mask.unsqueeze(-1).expand_as(mapping)
# Selecting the elements where mask is True
selected_mapping = mapping[expanded_mask].view(-1, 1024)
filtered_mapping = torch.unique(selected_mapping.view(-1))[
    torch.unique(selected_mapping.view(-1)) != -1
]
print(filtered_mapping.shape)
print(torch.max(filtered_mapping))
print(torch.min(filtered_mapping))

# move GSs
splats["means3d"][filtered_mapping, 0] = splats["means3d"][filtered_mapping, 0] + 0.5

render_image, features, info, meta = viewer_render_fn(
    camera_state=inputs["camera_state"],
    img_wh=inputs["img_wh"],  # (2048, 928)
    feature_query=query,
    feature_similarity_threshold=0.28,
    compute_mapping=False,
    segment=False,
)


plt.imsave("images/moved_image.png", render_image)
