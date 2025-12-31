import os
import sys
import numpy as np
from PIL import Image
import py360convert
import concurrent.futures
from tqdm import tqdm


def process_image(file, inputfolder, outputfolder, planar_size, planar_fov):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        return

    img_path = os.path.join(inputfolder, file)
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Failed to open {file}: {e}")
        return

    img_array = np.array(img)

    # Ensure image has 3 channels (RGB) for py360convert
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    try:
        rectilinear = py360convert.e2p(
            img_array,
            fov_deg=planar_fov,
            u_deg=0,
            v_deg=-90,
            out_hw=(planar_size, planar_size),
            in_rot_deg=0,
            mode='bilinear',
        )
    except Exception as e:
        print(f"Conversion failed for {file}: {e}")
        return

    # Ensure correct dtype
    if rectilinear.dtype != np.uint8:
        try:
            rectilinear = np.clip(rectilinear, 0, 255).astype(np.uint8)
        except Exception:
            rectilinear = rectilinear.astype(np.uint8)

    out_img = Image.fromarray(rectilinear)
    base_name = os.path.splitext(file)[0]
    out_path = os.path.join(outputfolder, base_name + '.png')
    try:
        out_img.save(out_path)
    except Exception as e:
        print(f"Failed to save {out_path}: {e}")


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert equirectangular images to planar top-down.")
    parser.add_argument("inputfolder", help="Input folder with equirectangular images")
    parser.add_argument("outputfolder", help="Output folder for planar images")
    # To maintain compatibility with positional arguments for planarSize, planarFov, threads
    parser.add_argument("planar_size", type=int, nargs="?", default=2048, help="Planar size (default 2048)")
    parser.add_argument("planar_fov", type=float, nargs="?", default=160.0, help="Planar FOV (default 160.0)")
    parser.add_argument("threads", type=int, nargs="?", default=None, help="Number of threads")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of frames to process")
    return parser.parse_args()

args = parse_args()
inputfolder = args.inputfolder
outputfolder = args.outputfolder
planar_size = args.planar_size
planar_fov = args.planar_fov
threads = args.threads
limit = args.limit

# create output folder if necessary
os.makedirs(outputfolder, exist_ok=True)

# Get list of image files
try:
    image_files = [f for f in os.listdir(inputfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    if limit > 0:
        image_files = image_files[:limit]
except Exception as e:
    print(f"Failed to list input folder '{inputfolder}': {e}")
    sys.exit(1)

# Process images in parallel
max_workers = threads if threads is not None else None
print(f"Found {len(image_files)} images. Converting...")
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(lambda f: process_image(f, inputfolder, outputfolder, planar_size, planar_fov), image_files), total=len(image_files)))
