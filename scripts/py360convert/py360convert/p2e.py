from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .utils import (
    Dim,
    DType,
    EquirecSampler,
    InterpolationMode,
    equirect_uvgrid,
    mode_to_order,
    rotation_matrix,
    uv2unitxyz,
)


def p2e(
    perspective_img: NDArray[DType],
    h: int,
    w: int,
    fov_deg: Union[float, int, tuple[Union[float, int], Union[float, int]]],
    u_deg: float,
    v_deg: float,
    in_rot_deg: float = 0,
    mode: InterpolationMode = "bilinear",
) -> NDArray[DType]:
    """Project a perspective image back onto an equirectangular sphere.

    Parameters
    ----------
    perspective_img: ndarray
        Perspective image in shape [H, W] or [H, W, C].
    h: int
        Height of the output equirectangular image.
    w: int
        Width of the output equirectangular image.
    fov_deg: scalar or tuple (scalar, scalar)
        Field of view of the perspective image in degrees.
    u_deg: float
        Horizontal viewing angle (yaw) in degrees.
    v_deg: float
        Vertical viewing angle (pitch) in degrees.
    in_rot_deg: float
        In-plane roll of the perspective camera in degrees.
    mode: str
        Interpolation mode.

    Returns
    -------
    np.ndarray
        Equirectangular panorama.
    """
    if perspective_img.ndim not in (2, 3):
        raise ValueError("perspective_img must have 2 or 3 dimensions.")

    squeeze = False
    if perspective_img.ndim == 2:
        perspective_img = perspective_img[..., None]
        squeeze = True

    if isinstance(fov_deg, (int, float, Real)):
        h_fov = v_fov = float(np.deg2rad(float(fov_deg)))
    else:
        h_fov = float(np.deg2rad(fov_deg[0]))
        v_fov = float(np.deg2rad(fov_deg[1]))

    if h_fov <= 0 or v_fov <= 0:
        raise ValueError("Field of view must be greater than zero.")

    yaw = -float(np.deg2rad(u_deg))
    pitch = float(np.deg2rad(v_deg))
    roll = float(np.deg2rad(in_rot_deg))

    Rx = rotation_matrix(pitch, Dim.X)
    Ry = rotation_matrix(yaw, Dim.Y)
    Ri_axis = np.array([0.0, 0.0, 1.0]).dot(Rx).dot(Ry)
    Ri = rotation_matrix(roll, Ri_axis)

    rotation = Rx.dot(Ry).dot(Ri)
    inv_rotation = rotation.T

    uu, vv = equirect_uvgrid(h, w)
    uv = np.stack([uu, vv], axis=-1)
    world_dirs = uv2unitxyz(uv)
    camera_dirs = world_dirs.dot(inv_rotation)

    denom = camera_dirs[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        plane_x = camera_dirs[..., 0] / denom
        plane_y = camera_dirs[..., 1] / denom

    max_x = np.tan(h_fov / 2)
    max_y = np.tan(v_fov / 2)

    mask = (
        np.isfinite(plane_x)
        & np.isfinite(plane_y)
        & (denom > 0)
        & (np.abs(plane_x) <= max_x)
        & (np.abs(plane_y) <= max_y)
    )

    plane_x = np.nan_to_num(plane_x, nan=0.0, posinf=0.0, neginf=0.0)
    plane_y = np.nan_to_num(plane_y, nan=0.0, posinf=0.0, neginf=0.0)

    source_h, source_w = perspective_img.shape[:2]
    normalized_x = (plane_x + max_x) / (2 * max_x)
    normalized_y = (max_y - plane_y) / (2 * max_y)

    coor_x = np.clip(normalized_x * (source_w - 1), 0, source_w - 1).astype(np.float32)
    coor_y = np.clip(normalized_y * (source_h - 1), 0, source_h - 1).astype(np.float32)
    coor_x = coor_x[..., None]
    coor_y = coor_y[..., None]

    order = mode_to_order(mode)
    sampler = EquirecSampler(coor_x, coor_y, order)

    projected = np.stack([sampler(perspective_img[..., i]) for i in range(perspective_img.shape[2])], axis=-1)

    if squeeze:
        projected = projected[..., 0]
    else:
        mask = mask[..., None]

    output = np.where(mask, projected, np.zeros_like(projected))
    return output
