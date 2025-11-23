import os
import sys
import numpy as np
from PIL import Image
import py360convert
import concurrent.futures


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
        print(f"Processed {file} -> {out_path}")
    except Exception as e:
        print(f"Failed to save {out_path}: {e}")


def _usage_and_exit():
    print("Usage: python 1_equirect_to_rectilinear.py <inputfolder> <outputfolder> [planarSize planarFov] [threads]")
    print("  planarSize: integer output width/height (default 2048)")
    print("  planarFov: field of view in degrees (default 160)")
    print("  threads: optional integer number of worker threads")
    sys.exit(1)


# Acceptable forms:
#  - script inputfolder outputfolder
#  - script inputfolder outputfolder planarSize planarFov
#  - script inputfolder outputfolder planarSize planarFov threads
if not (len(sys.argv) in (3, 5, 6)):
    _usage_and_exit()

inputfolder = sys.argv[1]
outputfolder = sys.argv[2]

# defaults
planar_size = 2048
planar_fov = 160.0
threads = None

if len(sys.argv) == 5 or len(sys.argv) == 6:
    try:
        planar_size = int(sys.argv[3])
        planar_fov = float(sys.argv[4])
    except ValueError:
        print("planarSize and planarFov must be numbers")
        _usage_and_exit()

if len(sys.argv) == 6:
    try:
        threads = int(sys.argv[5])
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

# Process images in parallel
if threads is None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda f: process_image(f, inputfolder, outputfolder, planar_size, planar_fov), image_files)
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(lambda f: process_image(f, inputfolder, outputfolder, planar_size, planar_fov), image_files)
