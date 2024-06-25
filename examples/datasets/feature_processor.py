from .clip import OpenCLIPNetwork, OpenCLIPNetworkConfig


import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert (
        False
    ), "open_clip is not installed, install it with `pip install open-clip-torch`"


class FeatureProcessor:
    def __init__(self, sam_ckpt: str, embed_dim: int = 256, device: str = "cuda"):
        self.embed_dim = embed_dim
        self.device = device

        self.model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        print("Feature Processor Initialized")

    def filter(self, keep: torch.Tensor, masks_result) -> None:
        keep = keep.int().cpu().numpy()
        result_keep = []
        for i, m in enumerate(masks_result):
            if i in keep:
                result_keep.append(m)
        return result_keep

    def mask_nms(
        self, masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs
    ):
        """
        Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

        Args:
            masks (torch.Tensor): has shape (num_masks, H, W)
            scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
            iou_thr (float, optional): The threshold for IoU.
            score_thr (float, optional): The threshold for the mask scores.
            inner_thr (float, optional): The threshold for the overlap rate.
            **kwargs: Additional keyword arguments.
        Returns:
            selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
        """

        scores, idx = scores.sort(0, descending=True)
        num_masks = idx.shape[0]

        masks_ord = masks[idx.view(-1), :]
        masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

        iou_matrix = torch.zeros(
            (num_masks,) * 2, dtype=torch.float, device=masks.device
        )
        inner_iou_matrix = torch.zeros(
            (num_masks,) * 2, dtype=torch.float, device=masks.device
        )
        for i in range(num_masks):
            for j in range(i, num_masks):
                intersection = torch.sum(
                    torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float
                )
                union = torch.sum(
                    torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float
                )
                iou = intersection / union
                iou_matrix[i, j] = iou
                # select mask pairs that may have a severe internal relationship
                if (
                    intersection / masks_area[i] < 0.5
                    and intersection / masks_area[j] >= 0.85
                ):
                    inner_iou = 1 - (intersection / masks_area[j]) * (
                        intersection / masks_area[i]
                    )
                    inner_iou_matrix[i, j] = inner_iou
                if (
                    intersection / masks_area[i] >= 0.85
                    and intersection / masks_area[j] < 0.5
                ):
                    inner_iou = 1 - (intersection / masks_area[j]) * (
                        intersection / masks_area[i]
                    )
                    inner_iou_matrix[j, i] = inner_iou

        iou_matrix.triu_(diagonal=1)
        iou_max, _ = iou_matrix.max(dim=0)
        inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
        inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
        inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
        inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

        keep = iou_max <= iou_thr
        keep_conf = scores > score_thr
        keep_inner_u = inner_iou_max_u <= 1 - inner_thr
        keep_inner_l = inner_iou_max_l <= 1 - inner_thr

        # If there are no masks with scores above threshold, the top 3 masks are selected
        if keep_conf.sum() == 0:
            index = scores.topk(3).indices
            keep_conf[index, 0] = True
        if keep_inner_u.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_u[index, 0] = True
        if keep_inner_l.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_l[index, 0] = True
        keep *= keep_conf
        keep *= keep_inner_u
        keep *= keep_inner_l

        selected_idx = idx[keep]
        return selected_idx

    def masks_update(self, *args, **kwargs):
        # remove redundant masks based on the scores and overlap rate between masks
        masks_new = ()
        for masks_lvl in args:
            seg_pred = torch.from_numpy(
                np.stack([m["segmentation"] for m in masks_lvl], axis=0)
            )
            iou_pred = torch.from_numpy(
                np.stack([m["predicted_iou"] for m in masks_lvl], axis=0)
            )
            stability = torch.from_numpy(
                np.stack([m["stability_score"] for m in masks_lvl], axis=0)
            )

            scores = stability * iou_pred
            keep_mask_nms = self.mask_nms(seg_pred, scores, **kwargs)
            masks_lvl = self.filter(keep_mask_nms, masks_lvl)

            masks_new += (masks_lvl,)
        return masks_new

    def get_seg_img(self, mask, image):
        image = image.copy()
        image[mask["segmentation"] == 0] = np.array([0, 0, 0], dtype=np.uint8)
        x, y, w, h = np.int32(mask["bbox"])
        seg_img = image[y : y + h, x : x + w, ...]
        return seg_img

    def pad_img(self, img):
        h, w, _ = img.shape
        l = max(w, h)
        pad = np.zeros((l, l, 3), dtype=np.uint8)
        if h > w:
            pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
        else:
            pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = img
        return pad

    def _sam_encoder(self, image):
        # pre-compute masks
        masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image)
        # pre-compute postprocess
        masks_default, masks_s, masks_m, masks_l = self.masks_update(
            masks_default,
            masks_s,
            masks_m,
            masks_l,
            iou_thr=0.8,
            score_thr=0.7,
            inner_thr=0.5,
        )

        def mask2segmap(masks, image):
            seg_img_list = []
            seg_map = -np.ones(image.shape[:2], dtype=np.int32)
            for i in range(len(masks)):
                mask = masks[i]
                seg_img = self.get_seg_img(mask, image)
                pad_seg_img = cv2.resize(self.pad_img(seg_img), (224, 224))
                seg_img_list.append(pad_seg_img)

                seg_map[masks[i]["segmentation"]] = i
            seg_imgs = np.stack(seg_img_list, axis=0)  # b,H,W,3
            seg_imgs = (
                torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
            ).to("cuda")

            return seg_imgs, seg_map

        seg_images, seg_maps = {}, {}
        seg_images["default"], seg_maps["default"] = mask2segmap(masks_default, image)
        if len(masks_s) != 0:
            seg_images["s"], seg_maps["s"] = mask2segmap(masks_s, image)
        if len(masks_m) != 0:
            seg_images["m"], seg_maps["m"] = mask2segmap(masks_m, image)
        if len(masks_l) != 0:
            seg_images["l"], seg_maps["l"] = mask2segmap(masks_l, image)

        return seg_images, seg_maps

    def _embed_clip_sam_tiles(self, image):
        seg_images, seg_map = self._sam_encoder(image)

        clip_embeds = {}
        for mode in ["default", "s", "m", "l"]:
            tiles = seg_images[mode]
            tiles = tiles.to("cuda")
            with torch.no_grad():
                clip_embed = self.model.encode_image(tiles)
            clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
            clip_embeds[mode] = clip_embed.detach().cpu().half()

        return clip_embeds, seg_map

    def process(self, image: np.ndarray):
        # image (H, W, 3)
        self.mask_generator.predictor.model.to(self.device)
        img_embed, seg_map = self._embed_clip_sam_tiles(image)
        for k, v in img_embed.items():
            print(k, v.shape)
        print("-" * 50)
        for k, v in seg_map.items():
            print(k, v.shape)
        print("-" * 50)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)

        print("total_length: ", total_length)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length

        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j - 1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j - 1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)

        print("img_embed", img_embed.shape)
        print("seg_map", seg_map.shape)

        # TODO: align features with image size then output embeddings.