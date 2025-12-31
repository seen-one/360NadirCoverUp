import os
import sys
import numpy as np
from PIL import Image
import concurrent.futures
import cv2
from tqdm import tqdm

# Import utils from py360convert for coordinate generation
from py360convert.utils import (
    Dim,
    rotation_matrix,
    equirect_uvgrid,
    uv2unitxyz,
)

def prepare_maps(equi_h, equi_w, fov_deg, u_deg, v_deg, in_rot_deg, src_h, src_w):
    """
    Pre-calculate look-up tables for remapping.
    Returns (map1, map2) compatible with cv2.remap.
    """
    print("Pre-calculating coordinate maps...")
    
    # Field of view
    h_fov = float(np.deg2rad(fov_deg))
    v_fov = h_fov # Assuming square pixels/aspect ratio for FOV

    # Rotation angles
    yaw = -float(np.deg2rad(u_deg))
    pitch = float(np.deg2rad(v_deg))
    roll = float(np.deg2rad(in_rot_deg))

    # Rotation matrix
    Rx = rotation_matrix(pitch, Dim.X)
    Ry = rotation_matrix(yaw, Dim.Y)
    Ri_axis = np.array([0.0, 0.0, 1.0]).dot(Rx).dot(Ry)
    Ri = rotation_matrix(roll, Ri_axis)

    rotation = Rx.dot(Ry).dot(Ri)
    inv_rotation = rotation.T

    # Generate Equirectangular grid
    uu, vv = equirect_uvgrid(equi_h, equi_w)
    uv = np.stack([uu, vv], axis=-1)
    
    # 3D Ray directions on the sphere
    world_dirs = uv2unitxyz(uv)
    
    # Rotate rays to camera local frame
    # Shape: (H, W, 3)
    camera_dirs = world_dirs.dot(inv_rotation)

    # Project to planar (Z is depth)
    # camera_dirs[..., 0] is X, [..., 1] is Y, [..., 2] is Z
    denom = camera_dirs[..., 2]
    
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        plane_x = camera_dirs[..., 0] / denom
        plane_y = camera_dirs[..., 1] / denom

    # Define valid frustum bounds
    max_x = np.tan(h_fov / 2)
    max_y = np.tan(v_fov / 2)

    # Validity mask
    # Z > 0 means in front of camera
    mask = (
        np.isfinite(plane_x)
        & np.isfinite(plane_y)
        & (denom > 0)
        & (np.abs(plane_x) <= max_x)
        & (np.abs(plane_y) <= max_y)
    )

    # Replace invalid values to avoid errors during normalization, 
    # though they will be masked out later by borderMode
    plane_x = np.nan_to_num(plane_x, nan=0.0, posinf=0.0, neginf=0.0)
    plane_y = np.nan_to_num(plane_y, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0, 1]
    normalized_x = (plane_x + max_x) / (2 * max_x)
    normalized_y = (max_y - plane_y) / (2 * max_y)

    # Scale to source image coordinates
    map_x = normalized_x * (src_w - 1)
    map_y = normalized_y * (src_h - 1)
    
    # Where mask is false, set coords to -1 (which will result in black border)
    map_x[~mask] = -1
    map_y[~mask] = -1

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Convert to fixed-point maps for faster remapping
    map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    
    return map1, map2

def process_image(file, inputfolder, outputfolder, map1, map2):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        return

    img_path = os.path.join(inputfolder, file)
    try:
        # Open using OpenCV for speed
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback to PIL if CV2 fails (e.g. some obscure format)
            pil_img = Image.open(img_path)
            img = np.array(pil_img)
            # Convert RGB to BGR for OpenCV if needed, but assuming standard flow
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to open {file}: {e}")
        return

    # Handle grayscale by converting to BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Determine border value based on channels (BGR or BGRA)
    if img.shape[2] == 4:
        bv = (0, 0, 0, 0)
    else:
        bv = (0, 0, 0)

    try:
        # Remap
        # INTER_LINEAR is bilinear interpolation
        # BORDER_CONSTANT with value 0/Alpha 0 means black/transparent background
        equirect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    except Exception as e:
        print(f"Conversion failed for {file}: {e}")
        return

    base_name = os.path.splitext(file)[0]
    out_path = os.path.join(outputfolder, base_name + '.png')
    
    try:
        cv2.imwrite(out_path, equirect)
    except Exception as e:
        print(f"Failed to save {out_path}: {e}")


def _usage_and_exit():
    print("Usage: python 4_planar_to_equirect.py <inputfolder> <outputfolder> [equiHeight equiWidth planarFov] [threads]")
    print("  equiHeight/equiWidth: output equirectangular size (default 2048 4096)")
    print("  planarFov: field of view of the perspective input in degrees (default 160)")
    print("  threads: optional integer number of worker threads")
    sys.exit(1)


# Acceptable forms:
#  - script inputfolder outputfolder
#  - script inputfolder outputfolder equiHeight equiWidth planarFov
#  - script inputfolder outputfolder equiHeight equiWidth planarFov threads
if not (len(sys.argv) in (3, 6, 7)):
    _usage_and_exit()

inputfolder = sys.argv[1]
outputfolder = sys.argv[2]

# defaults
equi_h = 2048
equi_w = 4096
planar_fov = 160.0
threads = None

if len(sys.argv) == 6 or len(sys.argv) == 7:
    try:
        equi_h = int(sys.argv[3])
        equi_w = int(sys.argv[4])
        planar_fov = float(sys.argv[5])
    except ValueError:
        print("equiHeight/equiWidth/planarFov must be numbers")
        _usage_and_exit()

if len(sys.argv) == 7:
    try:
        threads = int(sys.argv[6])
    except ValueError:
        print("threads must be an integer")
        _usage_and_exit()

# create output folder if necessary
os.makedirs(outputfolder, exist_ok=True)

# Get list of image files
try:
    image_files = [f for f in os.listdir(inputfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
except Exception as e:
    print(f"Failed to list input folder '{inputfolder}': {e}")
    sys.exit(1)

if not image_files:
    print("No images found to process.")
    sys.exit(0)

# Load the first image to determine source dimensions
first_img_path = os.path.join(inputfolder, image_files[0])
first_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
if first_img is None:
    # try pil
    first_img = np.array(Image.open(first_img_path))
    src_h, src_w = first_img.shape[:2]
else:
    src_h, src_w = first_img.shape[:2]

print(f"Reference image size: {src_w}x{src_h}")

# Pre-calculate maps once
# Using same yaw/pitch/roll as original: u=0, v=-90 (looking down), rot=0
map1, map2 = prepare_maps(
    equi_h, equi_w, 
    planar_fov, 
    u_deg=0, 
    v_deg=-90, 
    in_rot_deg=0, 
    src_h=src_h, 
    src_w=src_w
)

# Process images in parallel
# Since we are using OpenCV (C++), we can expect better parallelism even with threads
if threads is None:
    # Default to logical cores
    threads = os.cpu_count() or 4

print(f"Processing {len(image_files)} images using {threads} threads...")

with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
    # We use a lambda to pass the constant map arguments
    list(tqdm(executor.map(lambda f: process_image(f, inputfolder, outputfolder, map1, map2), image_files), total=len(image_files)))
