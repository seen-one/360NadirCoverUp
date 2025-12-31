import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import cv2
import numpy as np
from tqdm import tqdm


def recombine_image(original_path, overlay_path, out_path):
    ori = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    ov = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if ori is None:
        raise RuntimeError(f"Failed to read original image: {original_path}")
    if ov is None:
        raise RuntimeError(f"Failed to read overlay image: {overlay_path}")

    # Ensure 3-channel RGB for both (keep alpha if present on overlay)
    ori_rgb = ori[:, :, :3] if ori.ndim == 3 else cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)

    # Resize overlay to match original if necessary
    if (ov.shape[0], ov.shape[1]) != (ori_rgb.shape[0], ori_rgb.shape[1]):
        ov = cv2.resize(ov, (ori_rgb.shape[1], ori_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Determine overlay RGB and alpha mask
    if ov.ndim == 3 and ov.shape[2] == 4:
        ov_rgb = ov[:, :, :3]
        alpha = ov[:, :, 3].astype(np.float32) / 255.0
    else:
        ov_rgb = ov[:, :, :3] if ov.ndim == 3 else cv2.cvtColor(ov, cv2.COLOR_GRAY2BGR)
        # Derive alpha mask by difference from original
        diff = np.abs(ov_rgb.astype(np.int16) - ori_rgb.astype(np.int16))
        mask = (np.max(diff, axis=2) > 6).astype(np.float32)
        # Smooth mask a little to reduce hard edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        alpha = mask

    alpha_3 = np.stack([alpha, alpha, alpha], axis=2)

    blended = (ov_rgb.astype(np.float32) * alpha_3 + ori_rgb.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

    # If original had alpha, preserve it. Otherwise just write blended RGB
    if ori.ndim == 3 and ori.shape[2] == 4:
        out = np.dstack([blended, ori[:, :, 3]])
    else:
        out = blended

    # Write result (use PNG compression for lossless output)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".png", ".tif", ".tiff"]:
        cv2.imwrite(out_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(out_path, out)


def recombine_dir(orig_dir, overlay_dir, out_dir, workers=8, limit=0):
    os.makedirs(out_dir, exist_ok=True)

    # Supported image extensions
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

    # Build overlay lookup by relative path (without extension) and by basename (without extension)
    overlays_by_rel = {}
    overlays_by_base = {}
    for root, _, files in os.walk(overlay_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, overlay_dir)
            rel_no_ext = os.path.splitext(rel)[0]
            base_no_ext = os.path.splitext(f)[0]
            # prefer first occurrence for a given rel_no_ext
            overlays_by_rel.setdefault(rel_no_ext, full)
            overlays_by_base.setdefault(base_no_ext, []).append(full)

    # Scan originals only at top-level (do not recurse into subfolders)
    tasks = []
    orig_count = 0
    for p in glob.glob(os.path.join(orig_dir, "*")):
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in exts:
            continue
        orig_count += 1
        f = os.path.basename(p)
        orig_full = p
        rel = f  # only basename for top-level

        # First try exact relative path match (compare without extensions)
        rel_no_ext = os.path.splitext(rel)[0]
        overlay_full = overlays_by_rel.get(rel_no_ext)

        # Then try basename match (compare without extensions)
        if overlay_full is None:
            base_no_ext = os.path.splitext(f)[0]
            candidates = overlays_by_base.get(base_no_ext, [])
            if len(candidates) == 1:
                overlay_full = candidates[0]
            elif len(candidates) > 1:
                # Ambiguous: pick the first candidate (could be refined)
                overlay_full = candidates[0]

        if overlay_full:
            out_full = os.path.join(out_dir, rel)
            # out dir is flat for top-level originals
            tasks.append((orig_full, overlay_full, out_full))
    
    tasks.sort(key=lambda x: x[0])
    if workers > 0 and len(tasks) > 0:
        if "limit" in locals() and limit > 0:
            tasks = tasks[:limit]
        elif hasattr(args, 'limit') and args.limit > 0:
            tasks = tasks[:args.limit]

    if not tasks:
        print("No matching files found between original and overlay directories.")
        print(f"Original files scanned: {orig_count}")
        print(f"Overlay files available: {len(overlays_by_rel)}")
        sample_bases = list(overlays_by_base.keys())[:10]
        if sample_bases:
            print("Sample overlay basenames:", ", ".join(sample_bases))
        return

    print(f"Found {len(tasks)} matching images (scanned {orig_count} originals, {len(overlays_by_rel)} overlays)")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(recombine_image, *t): t for t in tasks}
        pbar = tqdm(total=len(futures), desc="Overlaying patches")
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error processing {t[0]} with overlay {t[1]}: {e}")
            pbar.update(1)
        pbar.close()


def parse_args():
    p = argparse.ArgumentParser(description="Overlay equirectangular patched nadir areas back onto original images.")
    p.add_argument("orig_dir", help="Directory with original images")
    p.add_argument("overlay_dir", help="Directory with overlay images (same basenames)")
    p.add_argument("out_dir", help="Output directory")
    # Accept either a named flag or an optional positional threads argument so callers
    # that pass the thread count without `--threads` (controller script) still work.
    p.add_argument("threads_pos", nargs="?", help="Optional positional threads (fallback if --threads not used)")
    p.add_argument("--threads", default=None, help="Number of worker processes (default: 8)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of frames to process")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Determine number of workers: prefer --threads, then positional, then default 8
    threads_val = None
    if args.threads is not None:
        threads_val = args.threads
    elif args.threads_pos is not None:
        threads_val = args.threads_pos

    try:
        workers = max(1, int(threads_val)) if threads_val is not None else 8
    except Exception:
        workers = 8

    recombine_dir(args.orig_dir, args.overlay_dir, args.out_dir, workers=workers, limit=args.limit)
