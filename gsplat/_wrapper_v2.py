from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonics.apply(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Normalized quaternions. [N, 4]
        scales: Scales. [N, 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple of:
        - Covariance matrices. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
        - Precision matrices. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
    """
    assert quats.dim() == 2 and quats.size(1) == 4, quats.size()
    assert scales.dim() == 2 and scales.size(1) == 3, scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, precis = _QuatScaleToCovarpreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple of:
        - Projected means. [C, N, 2]
        - Projected covariances. [C, N, 2, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _PerspProj.apply(means, covars, Ks, width, height)


def world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera space.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: Camera-to-world matrices. [C, 4, 4]

    Returns:
        A tuple of:
        - Gaussian means in camera space. [C, N, 3]
        - Gaussian covariances in camera space. [C, N, 3, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 3, 3), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _WorldToCam.apply(means, covars, viewmats)


def projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 6] or None
    quats: Tensor,  # [N, 4] or None
    scales: Tensor,  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    Note:
        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. `Radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6]
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the projected covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: The minimum radius of the projected Gaussians in pixel unit. Default: 1e10.

    Returns:
        A tuple of (if packed is True):
        - Rindices. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - Cindices. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - Radii. The maximum radius of the projected Gaussians in pixel unit.
            Int32 tensor of shape [nnz].
        - Projected means. [nnz, 2]
        - Depths. The z-depth of the projected Gaussians. [nnz]
        - Conics. Inverse of the projected covariances. Return the flattend upper
            triangle with [nnz, 3]

        A tuple of (if packed is False):
        - Radii. The maximum radius of the projected Gaussians in pixel unit.
            Int32 tensor of shape [C, N].
        - Projected means. [C, N, 2]
        - Depths. The z-depth of the projected Gaussians. [C, N]
        - Conics. Inverse of the projected covariances. Return the flattend upper
            triangle with [C, N, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _ProjectionPacked.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
        )
    else:
        return _Projection.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N] or [nnz]
    depths: Tensor,  # [C, N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    rindices: Optional[Tensor] = None,
    cindices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection
            ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        rindices: The row indices of the projected Gaussians. Required if packed is True.
        cindices: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple of:
        - Tiles per Gaussian. The number of tiles intersected by each Gaussian.
            Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - Intersection ids. Each id is an 64-bit integer with the following
            information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
            Xc and Xt are the maximum number of bits required to represent the camera
            and tile ids, respectively. Int64 [n_isects]
        - If pack, this is the indices in the nnz tensor. Else, this is the
            Gaussian ids. Int32 [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz,), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert rindices is not None, "rindices is required if packed is True"
        assert cindices is not None, "cindices is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        tiles_per_gauss, isect_ids, gauss_ids = _make_lazy_cuda_func(
            "isect_tiles_packed"
        )(
            n_cameras,
            rindices.contiguous(),
            cindices.contiguous(),
            means2d.contiguous(),
            radii.contiguous(),
            depths.contiguous(),
            tile_size,
            tile_width,
            tile_height,
            sort,
        )
    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N), radii.size()
        assert depths.shape == (C, N), depths.size()
        tiles_per_gauss, isect_ids, gauss_ids = _make_lazy_cuda_func("isect_tiles")(
            means2d.contiguous(),
            radii.contiguous(),
            depths.contiguous(),
            tile_size,
            tile_width,
            tile_height,
            sort,
        )
    return tiles_per_gauss, isect_ids, gauss_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    gauss_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    packed: bool = False,
    rindices: Optional[Tensor] = None,
    cindices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    C = isect_offsets.size(0)
    N = opacities.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert rindices is not None, "rindices is required if packed is True"
        assert cindices is not None, "cindices is required if packed is True"
    else:
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.empty(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if packed:
        render_colors, render_alphas = _RasterizeToPixelsPacked.apply(
            rindices.contiguous(),
            cindices.contiguous(),
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            backgrounds,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            gauss_ids.contiguous(),  # this is actually the pack_ids
        )
    else:
        render_colors, render_alphas = _RasterizeToPixels.apply(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            backgrounds,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            gauss_ids.contiguous(),
        )
    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


@torch.no_grad()
def rasterize_to_indices_iter(
    step0: int,
    step1: int,
    transmittances: Tensor,  # [C, image_height, image_width]
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    gauss_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor]:
    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (N,), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_pixel_ids = _make_lazy_cuda_func("rasterize_to_indices_iter")(
        step0,
        step1,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        gauss_ids.contiguous(),
    )
    return out_gauss_ids, out_pixel_ids


class _QuatScaleToCovarpreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4],
        scales: Tensor,  # [N, 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
            triu,
        )
        return v_quats, v_scales, None, None, None


class _PerspProj(torch.autograd.Function):
    """Perspective projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor]:
        means2d, covars2d = _make_lazy_cuda_func("persp_proj_fwd")(
            means, covars, Ks, width, height
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means, v_covars = _make_lazy_cuda_func("persp_proj_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None


class _WorldToCam(torch.autograd.Function):
    """Transforms Gaussians from world to camera space."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 3, 3]
        viewmats: Tensor,  # [C, 4, 4]
    ) -> Tuple[Tensor, Tensor]:
        means_c, covars_c = _make_lazy_cuda_func("world_to_cam_fwd")(
            means, covars, viewmats
        )
        ctx.save_for_backward(means, covars, viewmats)
        return means_c, covars_c

    @staticmethod
    def backward(ctx, v_means_c: Tensor, v_covars_c: Tensor):
        means, covars, viewmats = ctx.saved_tensors
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("world_to_cam_bwd")(
            means,
            covars,
            viewmats,
            v_means_c.contiguous(),
            v_covars_c.contiguous(),
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats


class _Projection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics = _make_lazy_cuda_func("projection_fwd")(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(means, covars, quats, scales, viewmats, Ks, radii, conics)
        ctx.width = width
        ctx.height = height

        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics):
        means, covars, quats, scales, viewmats, Ks, radii, conics = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            radii,
            conics,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [N]
        backgrounds: Tensor,  # [C, D], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        gauss_ids: Tensor,  # [n_isects]
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            gauss_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
        )

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
        )


class _ProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            rindices,
            cindices,
            radii,
            means2d,
            depths,
            conics,
        ) = _make_lazy_cuda_func("projection_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            rindices, cindices, means, covars, quats, scales, viewmats, Ks, conics
        )
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad

        return rindices, cindices, radii, means2d, depths, conics

    @staticmethod
    def backward(ctx, v_rindices, v_cindices, v_radii, v_means2d, v_depths, v_conics):
        (
            rindices,
            cindices,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        sparse_grad = ctx.sparse_grad
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            rindices,
            cindices,
            conics,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: this is somehow consuming more memory than expected!
                # Also cindices is duplicated so not idea.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=cindices[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=cindices[None],  # [1, nnz]
                    values=v_covars,  # [nnz, 6]
                    size=covars.size(),  # [N, 6]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=cindices[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                ).to_dense()  # TODO: F.normalize is preventing sparse gradients
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=cindices[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from .cuda_v2._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


class _RasterizeToPixelsPacked(torch.autograd.Function):
    """Rasterize gaussians packed"""

    @staticmethod
    def forward(
        ctx,
        rindices: Tensor,  # [nnz]
        cindices: Tensor,  # [nnz]
        means2d: Tensor,  # [nnz, 2]
        conics: Tensor,  # [nnz, 3]
        colors: Tensor,  # [nnz, 3]
        opacities: Tensor,  # [N]
        backgrounds: Tensor,  # [C, 3] Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        pack_ids: Tensor,  # [n_isects]
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_packed_fwd"
        )(
            rindices,
            cindices,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            pack_ids,
        )

        ctx.save_for_backward(
            rindices,
            cindices,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            isect_offsets,
            pack_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
    ):
        (
            rindices,
            cindices,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            isect_offsets,
            pack_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        (
            v_means2d_norm,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_packed_bwd")(
            rindices,
            cindices,
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            pack_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
        )

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        means2d.normgrad = v_means2d_norm

        return (
            None,
            None,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
        )


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("compute_sh_fwd")(sh_degree, dirs, coeffs, masks)
        ctx.save_for_backward(dirs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        num_bases = ctx.num_bases
        v_coeffs = _make_lazy_cuda_func("compute_sh_bwd")(
            num_bases, sh_degree, dirs, masks, v_colors.contiguous()
        )
        return None, None, v_coeffs, None