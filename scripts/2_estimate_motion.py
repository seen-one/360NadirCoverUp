import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def list_image_files(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    return files


def load_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")
    h, w = shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Mask semantics: non-zero pixels are excluded areas.
    return mask


def detect_features(img_gray: np.ndarray, allow_mask: np.ndarray) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(
        img_gray,
        mask=allow_mask,
        maxCorners=2000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        useHarrisDetector=False,
        k=0.04,
    )
    return pts


def estimate_pair_motion_affine(prev_gray: np.ndarray, next_gray: np.ndarray, exclude_mask: np.ndarray):
    # Build allowed mask (1 where allowed)
    allow_mask = (exclude_mask == 0).astype(np.uint8) * 255
    prev_pts = detect_features(prev_gray, allow_mask)
    if prev_pts is None or len(prev_pts) < 6:
        return None

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)
    if next_pts is None:
        return None

    status = status.reshape(-1)
    prev_good = prev_pts[status == 1]
    next_good = next_pts[status == 1]

    if len(prev_good) < 6:
        return None

    M, inliers = cv2.estimateAffinePartial2D(prev_good, next_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None

    # AffinePartial2D gives matrix [[a, b, tx],[c, d, ty]] where rotation+scale encoded.
    a, b, tx = M[0]
    c, d, ty = M[1]
    # Scale and rotation extraction: rotation angle = atan2(c, a)
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    # Approximate uniform scale
    scale = (scale_x + scale_y) / 2.0
    rot_rad = math.atan2(c, a)
    rot_deg = math.degrees(rot_rad)
    return {
        "dx": tx,
        "dy": ty,
        "rotation_deg": rot_deg,
        "scale": scale,
        "num_tracked": len(prev_good),
        "num_inliers": int(inliers.sum()) if inliers is not None else len(prev_good),
    }


def estimate_pair_motion_homography(prev_gray: np.ndarray, next_gray: np.ndarray, exclude_mask: np.ndarray):
    # Use features outside the mask, track with LK and estimate homography via RANSAC
    allow_mask = (exclude_mask == 0).astype(np.uint8) * 255
    prev_pts = detect_features(prev_gray, allow_mask)
    if prev_pts is None or len(prev_pts) < 8:
        return None

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)
    if next_pts is None:
        return None

    status = status.reshape(-1)
    prev_good = prev_pts[status == 1].reshape(-1, 2)
    next_good = next_pts[status == 1].reshape(-1, 2)

    if len(prev_good) < 8:
        return None

    H, inliers = cv2.findHomography(prev_good, next_good, cv2.RANSAC, 3.0)
    if H is None:
        return None

    # Map image center to estimate translation and approximate rotation
    h0, w0 = prev_gray.shape[:2]
    cx = w0 / 2.0
    cy = h0 / 2.0
    src = np.array([[[cx, cy]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    dx = float(dst[0, 0, 0] - cx)
    dy = float(dst[0, 0, 1] - cy)

    a = H[0, 0]
    b = H[0, 1]
    c = H[1, 0]
    d = H[1, 1]
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    scale = (scale_x + scale_y) / 2.0
    rot_rad = math.atan2(c, a)
    rot_deg = math.degrees(rot_rad)

    return {
        "dx": dx,
        "dy": dy,
        "rotation_deg": rot_deg,
        "scale": scale,
        "num_tracked": len(prev_good),
        "num_inliers": int(inliers.sum()) if inliers is not None else len(prev_good),
        "H": H,
    }


def estimate_pair_motion_dense(prev_gray: np.ndarray, next_gray: np.ndarray, exclude_mask: np.ndarray):
    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # Exclude masked pixels: mark them NaN
    ex = exclude_mask > 0
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    flow_x[ex] = np.nan
    flow_y[ex] = np.nan
    # Global translation as median of allowed flow
    dx = np.nanmedian(flow_x)
    dy = np.nanmedian(flow_y)
    # Estimate rotation by looking at angle change over a grid? Simplify: use gradients of flow (small motion assumption)
    # d(flow)/d(x,y) for rotation ~ (dvy/dx - dvx/dy)/2 ; we approximate using central differences ignoring NaNs.
    def safe_grad(arr):
        arr_filled = np.where(np.isnan(arr), 0, arr)
        gx = cv2.Sobel(arr_filled, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr_filled, cv2.CV_64F, 0, 1, ksize=3)
        return gx, gy
    dvx_dx, dvx_dy = safe_grad(flow_x)
    dvy_dx, dvy_dy = safe_grad(flow_y)
    rot_approx = (dvy_dx - dvx_dy) / 2.0
    rotation_deg = math.degrees(np.nanmedian(rot_approx) * 0.01)  # heuristic scaling
    return {
        "dx": float(dx),
        "dy": float(dy),
        "rotation_deg": float(rotation_deg),
        "scale": 1.0,
        "num_tracked": int(np.isfinite(flow_x).sum()),
        "num_inliers": int(np.isfinite(flow_x).sum()),
    }


def process_sequence(frames: List[Path], mask_path: Path, method: str, size: int = 0) -> List[dict]:
    results = []
    if len(frames) < 2:
        return results
    first = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Failed to read frame {frames[0]}")
    # If `size` is provided (>0) we will resize frames and the mask to (size, size).
    # If size == 0 (default) we process at original resolution.
    orig_h, orig_w = first.shape[:2]
    if size is None:
        size = 0
    if size <= 0:
        # No resizing
        scaled_shape = (orig_h, orig_w)
        sx = sy = 1.0
        mask = load_mask(mask_path, scaled_shape)
        prev_gray = first
    else:
        scaled_h = int(size)
        scaled_w = int(size)
        scaled_shape = (scaled_h, scaled_w)
        # scale factors from original -> scaled
        sx = scaled_w / float(orig_w)
        sy = scaled_h / float(orig_h)
        mask = load_mask(mask_path, scaled_shape)
        prev_gray = cv2.resize(first, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    cumulative_rot = 0.0

    for i in range(1, len(frames)):
        next_gray = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if next_gray is None:
            continue
        # Resize next frame to the processing size if needed
        if next_gray.shape[:2] != scaled_shape:
            next_gray = cv2.resize(next_gray, (scaled_shape[1], scaled_shape[0]), interpolation=cv2.INTER_AREA)

        if method == "affine":
            motion = estimate_pair_motion_affine(prev_gray, next_gray, mask)
        elif method == "homography":
            motion = estimate_pair_motion_homography(prev_gray, next_gray, mask)
        else:
            motion = estimate_pair_motion_dense(prev_gray, next_gray, mask)

        if motion is None:
            motion = {"dx": np.nan, "dy": np.nan, "rotation_deg": np.nan, "scale": np.nan, "num_tracked": 0, "num_inliers": 0}
        else:
            # Convert translation results from processed-image pixels back to original-image pixels
            if size <= 0:
                # No conversion needed
                if not math.isnan(motion.get("dx", float("nan"))):
                    cumulative_dx += motion["dx"]
                if not math.isnan(motion.get("dy", float("nan"))):
                    cumulative_dy += motion["dy"]
            else:
                if not math.isnan(motion.get("dx", float("nan"))):
                    # dx_scaled = dx_original * sx  => dx_original = dx_scaled / sx
                    motion["dx"] = motion["dx"] / sx
                    cumulative_dx += motion["dx"]
                if not math.isnan(motion.get("dy", float("nan"))):
                    motion["dy"] = motion["dy"] / sy
                    cumulative_dy += motion["dy"]
            if not math.isnan(motion.get("rotation_deg", float("nan"))):
                cumulative_rot += motion["rotation_deg"]

        # If homography was estimated attach flattened H elements
        h_fields = {}
        # If homography was estimated attach flattened H elements, converting to original coords if resized
        if "H" in motion and isinstance(motion["H"], np.ndarray):
            H = motion.pop("H")
            if size > 0:
                # S maps original -> scaled: [sx 0 0; 0 sy 0; 0 0 1]
                S = np.diag([sx, sy, 1.0])
                S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
                H = S_inv.dot(H).dot(S)
            for r_i in range(3):
                for c_i in range(3):
                    h_fields[f"h{r_i}{c_i}"] = float(H[r_i, c_i])

        record = {
            "frame_index": i,
            "frame_name": frames[i].name,
            **motion,
            **h_fields,
            "cumulative_dx": cumulative_dx,
            "cumulative_dy": cumulative_dy,
            "cumulative_rotation_deg": cumulative_rot,
        }
        results.append(record)
        prev_gray = next_gray
    return results


def write_csv(records: List[dict], out_path: Path):
    if not records:
        print("No motion records to write.")
        return
    fieldnames = list(records[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} motion rows to {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Estimate motion between consecutive frames excluding a masked region.")
    ap.add_argument("--frames_dir", required=True, help="Directory containing ordered frame images")
    ap.add_argument("--mask_path", required=True, help="Path to mask image (non-zero pixels excluded)")
    ap.add_argument("--method", choices=["affine", "dense", "homography"], default="affine", help="Motion estimation method")
    ap.add_argument("--output_csv", default="motion.csv", help="Output CSV path")
    ap.add_argument("--size", type=int, default=0, help="Fixed size (px) to resize both width and height before estimation (e.g. 512). Use 0 to keep original size.")
    return ap.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    mask_path = Path(args.mask_path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    frames = list_image_files(frames_dir)
    if len(frames) < 2:
        print("Need at least two frames to estimate motion.")
        return
    print(f"Found {len(frames)} frames. Using method '{args.method}'. Size={args.size}")
    records = process_sequence(frames, mask_path, args.method, args.size)
    write_csv(records, Path(args.output_csv))


if __name__ == "__main__":
    main()
