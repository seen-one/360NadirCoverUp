import argparse
from pathlib import Path
import cv2
import numpy as np
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

"""Fill masked pixels in each frame by warping adjacent frames (using homographies) into target coordinates
and selecting pixels from neighbors where the mask is not present.

Requirements:
- A motion CSV produced by `estimate_motion.py --method homography` containing h00..h22 fields per row.
- A single `mask.png` (mask is assumed in camera/image coords indicating pixels to cover).

Algorithm:
- Build cumulative homographies cum_H[k] mapping frame0 -> frame_k using the per-step H increments from CSV.
- For target frame i, for each neighbor j in window [i-w .. i+w] (j != i):
    - compute H_j_to_i = cum_H_i @ inv(cum_H_j)
    - warp frame_j into frame_i coordinates
    - warp neighbor's mask (same mask image) to mark pixels that are valid (i.e., where neighbor's mask == 0)
- For each masked pixel in frame_i, collect candidate pixels from warped neighbors where neighbor mask allows it.
  - Choose pixel from nearest neighbor (smallest |j-i|) or median across candidates (mode=`nearest` or `median`).
- Output filled frames to an output directory and a short coverage report.
"""


def read_motion_homographies(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    H_list = []
    for r in rows:
        if all(f"h{ri}{ci}" in r and r[f"h{ri}{ci}"] != "" for ri in range(3) for ci in range(3)):
            H = np.eye(3, dtype=np.float64)
            for ri in range(3):
                for ci in range(3):
                    H[ri, ci] = float(r[f"h{ri}{ci}"])
            H_list.append(H)
        else:
            H_list.append(None)
    return H_list


def list_frames(frames_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]


def build_cumulative_H(H_increments, n_frames):
    # H_increments corresponds to mapping frame[k] -> frame[k+1] for k=0..n-2
    cum_H = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]
    cur = np.eye(3, dtype=np.float64)
    for k in range(1, n_frames):
        H_inc = H_increments[k-1] if (k-1) < len(H_increments) else None
        if H_inc is None:
            H_inc = np.eye(3, dtype=np.float64)
        # cur maps frame0 -> frame_{k-1}; after update cur = H_inc @ cur maps frame0 -> frame_k
        cur = H_inc @ cur
        cum_H[k] = cur.copy()
    return cum_H


def estimate_global_camera_heading_from_increments(H_increments, img_w, img_h):
    """Estimate an average camera heading (degrees) from H increments by mapping image center
    through each H_inc and averaging displacement vectors. Returns angle in degrees.
    If estimation fails returns None.
    """
    if not H_increments:
        return None
    center = np.array([img_w / 2.0, img_h / 2.0, 1.0], dtype=np.float64)
    vecs = []
    for H in H_increments:
        if H is None:
            continue
        p = H @ center
        if p[2] == 0:
            continue
        p = p / p[2]
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        vecs.append((dx, dy))
    if not vecs:
        return None
    avg_dx = float(np.mean([v[0] for v in vecs]))
    avg_dy = float(np.mean([v[1] for v in vecs]))
    # heading: positive x to the right, positive y down; use -dy so forward is upward in image coords
    heading_rad = math.atan2(-avg_dy, avg_dx)
    return math.degrees(heading_rad)


def estimate_heading_from_H(H, img_w, img_h):
    """Estimate camera heading (degrees) from a single homography matrix H.
    H should map frame_k -> frame_{k+1}. The heading is the direction of motion.
    Returns angle in degrees, or None if H is invalid.
    """
    if H is None:
        return None
    center = np.array([img_w / 2.0, img_h / 2.0, 1.0], dtype=np.float64)
    p = H @ center
    if p[2] == 0:
        return None
    p = p / p[2]
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    heading_rad = math.atan2(-dy, dx)
    return math.degrees(heading_rad)

def fill_frames(
    frames_dir: Path,
    mask_path: Path,
    motion_csv: Path,
    out_dir: Path,
    window: int,
    method: str,
    donor_side: str = "both",
    sun_azimuth: float = None,
    feather: int = 0,
    transparent: bool = False,
    patch_smooth: int = 0,
    threads: int = None,
    coverage_report: bool = False,
):
    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError("No frames found")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read homography increments from CSV
    H_increments = read_motion_homographies(motion_csv)
    n = len(frames)
    cum_H = build_cumulative_H(H_increments, n)

    # load mask (single image assumed); mask non-zero == masked/occluded
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Failed to read mask {mask_path}")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    if mask_img.shape[:2] != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
    # Preserve the mask's per-pixel alpha (0-255) and a normalized float version (0.0-1.0)
    mask_alpha = mask_img.astype(np.uint8)
    mask_alpha_f = mask_alpha.astype(np.float32) / 255.0
    # Boolean mask used for detecting locations to fill (non-zero in mask means masked/occluded)
    mask_bool = mask_alpha > 0

    feather_alpha = None
    if feather and feather > 0:
        mask_uint8 = mask_bool.astype(np.uint8)
        dist_inside = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        feather_alpha = np.clip(dist_inside / float(feather), 0.0, 1.0)

    coverage_data = []

    # Preload frames to speed up warping
    imgs = [cv2.imread(str(p)) for p in frames]

    def process_frame(i):
        """Process a single frame - returns (i, target, filled, num_masked, frame_name, donor_side, sun_relative_angle)"""
        initial_donor_side = donor_side
        sun_relative_angle = None
        if transparent:
            # Start with transparent background (BGRA)
            target = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            target = imgs[i].copy()
        masked_positions = np.where(mask_bool)
        num_masked = int(mask_bool.sum())
        filled = 0

        # Prepare an accumulator for median if requested
        if method == "median":
            candidates = np.zeros((num_masked, 3, 0), dtype=np.uint8)

        # For nearest, we stop when we find a candidate per pixel
        filled_pixels = np.zeros_like(mask_bool, dtype=bool)
        # Create array to hold chosen pixels
        chosen = np.zeros((h, w, 3), dtype=np.uint8)
        # Track which donor frame each pixel came from (for patch smoothing)
        donor_id = np.full((h, w), -1, dtype=np.int32)

        # If window <= 0 interpret as searching the entire sequence
        max_dist = window if window > 0 else max(i, n - 1 - i)
        # determine neighbor sign order.

        # If donor_side is 'auto', determine it per-frame based on current motion
        if initial_donor_side == "auto" and sun_azimuth is not None:
            # Use the homography for the transition from this frame to the next
            H_inc = H_increments[i] if i < len(H_increments) else None
            cam_heading = estimate_heading_from_H(H_inc, w, h)

            if cam_heading is not None:
                # normalize angular difference to [-180, 180]
                diff = (((sun_azimuth - cam_heading + 180) % 360) - 180)
                sun_relative_angle = diff
                if abs(diff) <= 90:
                    current_donor_side = "front"
                else:
                    current_donor_side = "back"
            else:
                # Fallback if heading can't be estimated for this frame
                current_donor_side = "both"
        else:
            current_donor_side = initial_donor_side

        # If donor_side is 'front' or 'back', restrict donors to that side only.
        if current_donor_side == "front":
            pref_sign_order = (1,)
            other_sign_order = (-1,)
        elif current_donor_side == "back":
            pref_sign_order = (-1,)
            other_sign_order = (1,)
        else:
            # donor_side == 'both' -> default ordering prefers previous frame first
            pref_sign_order = (-1, 1)
            other_sign_order = ()

        for dist in range(1, max_dist + 1):
            for sign in pref_sign_order:
                j = i + sign * dist
                if j < 0 or j >= n:
                    continue
                H_j = cum_H[j]
                H_i = cum_H[i]
                try:
                    inv_H_j = np.linalg.inv(H_j)
                except np.linalg.LinAlgError:
                    inv_H_j = np.eye(3, dtype=np.float64)
                H_j_to_i = H_i @ inv_H_j

                # Warp neighbor image into target coordinates
                warped = cv2.warpPerspective(imgs[j], H_j_to_i, (w, h), flags=cv2.INTER_LINEAR)
                # Warp neighbor mask (where neighbor is occluded) to target coordinates
                warped_mask = cv2.warpPerspective(mask_img, H_j_to_i, (w, h), flags=cv2.INTER_NEAREST)
                neighbor_valid = warped_mask == 0

                if method == "nearest":
                    need_fill = (~filled_pixels) & mask_bool & neighbor_valid
                    if need_fill.any():
                        chosen[need_fill] = warped[need_fill]
                        filled_pixels[need_fill] = True
                        donor_id[need_fill] = j
                        filled += int(need_fill.sum())
                else:  # median
                    # collect candidate pixels for masked locations where neighbor_valid
                    # expand candidate arrays as needed
                    # we will store candidates as lists per color channel; easier to collect in list and compute median later
                    pass
            if method == "nearest" and filled == num_masked:
                break

        # If a single-side donor was requested and not all pixels were filled, try the other side as a fallback
        if method == "nearest" and filled < num_masked and other_sign_order:
            for dist in range(1, max_dist + 1):
                for sign in other_sign_order:
                    j = i + sign * dist
                    if j < 0 or j >= n:
                        continue
                    H_j = cum_H[j]
                    H_i = cum_H[i]
                    try:
                        inv_H_j = np.linalg.inv(H_j)
                    except np.linalg.LinAlgError:
                        inv_H_j = np.eye(3, dtype=np.float64)
                    H_j_to_i = H_i @ inv_H_j

                    warped = cv2.warpPerspective(imgs[j], H_j_to_i, (w, h), flags=cv2.INTER_LINEAR)
                    warped_mask = cv2.warpPerspective(mask_img, H_j_to_i, (w, h), flags=cv2.INTER_NEAREST)
                    neighbor_valid = warped_mask == 0

                    need_fill = (~filled_pixels) & mask_bool & neighbor_valid
                    if need_fill.any():
                        chosen[need_fill] = warped[need_fill]
                        filled_pixels[need_fill] = True
                        donor_id[need_fill] = j
                        filled += int(need_fill.sum())
                if filled == num_masked:
                    break

        if method == "median":
            # If median method selected, do a second pass to collect pixel stacks
            stacks = [[[] for _ in range(3)] for _ in range(num_masked)]
            idxs = list(zip(masked_positions[0].tolist(), masked_positions[1].tolist()))
            # For each neighbor in increasing temporal distance collect pixels
            def collect_stacks_for_signs(signs, collect_mask=None):
                # collect_mask: boolean array of length num_masked indicating which masked pixels still need candidates
                for dist in range(1, window + 1):
                    for sign in signs:
                        j = i + sign * dist
                        if j < 0 or j >= n:
                            continue
                        H_j = cum_H[j]
                        H_i = cum_H[i]
                        try:
                            inv_H_j = np.linalg.inv(H_j)
                        except np.linalg.LinAlgError:
                            inv_H_j = np.eye(3, dtype=np.float64)
                        H_j_to_i = H_i @ inv_H_j
                        warped = cv2.warpPerspective(imgs[j], H_j_to_i, (w, h), flags=cv2.INTER_LINEAR)
                        warped_mask = cv2.warpPerspective(mask_img, H_j_to_i, (w, h), flags=cv2.INTER_NEAREST)
                        neighbor_valid = warped_mask == 0
                        for k, (rr, cc) in enumerate(idxs):
                            if collect_mask is not None and not collect_mask[k]:
                                continue
                            if neighbor_valid[rr, cc]:
                                pix = warped[rr, cc]
                                stacks[k][0].append(int(pix[0]))
                                stacks[k][1].append(int(pix[1]))
                                stacks[k][2].append(int(pix[2]))

            # First collect from preferred side
            collect_stacks_for_signs(pref_sign_order)
            # compute median where stacks not empty (preferred side)
            remaining = [True] * num_masked
            for k, (rr, cc) in enumerate(idxs):
                r0, g0, b0 = stacks[k]
                if r0:
                    mv = [int(np.median(r0)), int(np.median(g0)), int(np.median(b0))]
                    chosen[rr, cc] = mv
                    filled += 1
                    remaining[k] = False

            # If not fully filled and an other side exists, collect additional candidates for remaining pixels
            if any(remaining) and other_sign_order:
                collect_mask = remaining
                collect_stacks_for_signs(other_sign_order, collect_mask=collect_mask)
                for k, (rr, cc) in enumerate(idxs):
                    if not remaining[k]:
                        continue
                    r0, g0, b0 = stacks[k]
                    if r0:
                        mv = [int(np.median(r0)), int(np.median(g0)), int(np.median(b0))]
                        chosen[rr, cc] = mv
                        filled += 1
                        remaining[k] = False
        # Apply patch smoothing if requested (for nearest method)
        if method == "nearest" and patch_smooth > 0 and filled > 0:
            # For each unique donor frame, create a distance transform from patch boundaries
            unique_donors = np.unique(donor_id[donor_id >= 0])
            if len(unique_donors) > 1:
                # Build alpha blending weights based on distance to patch boundaries
                blend_weights = np.zeros((h, w, len(unique_donors)), dtype=np.float32)
                for idx, donor in enumerate(unique_donors):
                    donor_mask = (donor_id == donor).astype(np.uint8)
                    # Distance transform from edges of this patch
                    dist = cv2.distanceTransform(donor_mask, cv2.DIST_L2, 5)
                    # Convert to alpha weight (0 at edge, 1 at patch_smooth distance inward)
                    weight = np.clip(dist / float(patch_smooth), 0.0, 1.0)
                    # Apply Gaussian blur to smooth the weight transitions
                    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=patch_smooth/3.0)
                    blend_weights[:, :, idx] = weight
                
                # Normalize weights so they sum to 1 at each pixel
                weight_sum = blend_weights.sum(axis=2, keepdims=True)
                weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
                blend_weights = blend_weights / weight_sum
                
                # Re-warp each donor frame and blend with weights
                blended_chosen = np.zeros((h, w, 3), dtype=np.float32)
                for idx, donor in enumerate(unique_donors):
                    # Re-warp the donor frame
                    H_j = cum_H[int(donor)]
                    H_i = cum_H[i]
                    try:
                        inv_H_j = np.linalg.inv(H_j)
                    except np.linalg.LinAlgError:
                        inv_H_j = np.eye(3, dtype=np.float64)
                    H_j_to_i = H_i @ inv_H_j
                    warped_donor = cv2.warpPerspective(imgs[int(donor)], H_j_to_i, (w, h), flags=cv2.INTER_LINEAR)
                    
                    # Apply weight for this donor
                    weight = blend_weights[:, :, idx:idx+1]
                    blended_chosen += weight * warped_donor.astype(np.float32)
                
                chosen = blended_chosen.astype(np.uint8)
        
        # Apply chosen pixels to target, optionally feathered at mask edges
        if method == "nearest" or method == "median":
            if method == "nearest":
                filled_mask = filled_pixels
            else:
                filled_mask = chosen.any(axis=2)
            apply_mask = mask_bool & filled_mask

            if transparent:
                # Place chosen RGB values into the BGRA target
                target[apply_mask, :3] = chosen[apply_mask]
                # Compute per-pixel alpha from the original mask values, optionally modulated by feather
                base_alpha = mask_alpha_f
                if feather_alpha is not None:
                    final_alpha = (base_alpha * feather_alpha).clip(0.0, 1.0)
                else:
                    final_alpha = base_alpha
                # Assign alpha only for the applied pixels (scale to 0-255)
                target[apply_mask, 3] = (final_alpha[apply_mask] * 255.0).astype(np.uint8)
            else:
                if feather_alpha is not None:
                    alpha = feather_alpha[..., None]
                    blended = (
                        alpha * chosen.astype(np.float32)
                        + (1.0 - alpha) * imgs[i].astype(np.float32)
                    ).astype(np.uint8)
                    target[apply_mask] = blended[apply_mask]
                else:
                    target[apply_mask] = chosen[apply_mask]

        return (i, target, filled, num_masked, frames[i].name, current_donor_side, sun_relative_angle)

    # Process frames in parallel using multithreading
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_frame, i): i for i in range(n)}
        for future in as_completed(futures):
            i, target, filled, num_masked, frame_name, side_used, sun_angle = future.result()
            out_path = out_dir / frame_name
            # If transparent output requested, ensure we save as PNG so alpha is preserved
            if transparent and out_path.suffix.lower() != ".png":
                out_path = out_path.with_suffix('.png')
            cv2.imwrite(str(out_path), target)
            if coverage_report:
                coverage_data.append({"frame": frame_name, "masked_pixels": int(num_masked), "filled_pixels": int(filled), "side_used": side_used, "sun_angle_rel": f"{sun_angle:.1f}" if sun_angle is not None else ""})
            
            angle_str = f", sun_angle: {sun_angle:6.1f}°" if sun_angle is not None else ""
            print(f"Frame {i}/{n-1}: filled {filled}/{num_masked} (side: {side_used}){angle_str}")
    
    # write coverage CSV if requested
    if coverage_report:
        # Sort coverage report by frame number
        coverage_data.sort(key=lambda x: x["frame"])
        with (out_dir / "coverage_report.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "masked_pixels", "filled_pixels", "side_used", "sun_angle_rel"])
            writer.writeheader()
            writer.writerows(coverage_data)
        print(f"Filled frames written to {out_dir}; report saved as coverage_report.csv")
    else:
        print(f"Filled frames written to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Fill masked pixels by warping neighboring frames into target coordinates using homographies.")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--motion_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=5, help="Maximum number of frames before/after to search")
    ap.add_argument("--method", choices=["nearest", "median"], default="nearest")
    ap.add_argument("--donor_side", choices=["both", "front", "back", "auto"], default="both", help="Restrict donor frames to one side: 'front' (later), 'back' (earlier), 'both', or 'auto' to pick based on sun azimuth (per-frame)")
    ap.add_argument("--sun_azimuth", type=float, help="Sun azimuth in degrees (0 = camera forward). Used when --donor_side auto.")
    ap.add_argument("--feather", type=int, default=0, help="Feather width in pixels for blending mask edges")
    ap.add_argument("--transparent", action="store_true", help="Output only filled pixels on transparent background (PNG)")
    ap.add_argument("--patch_smooth", type=int, default=0, help="Smoothing width in pixels for blending seams between donor patches")
    ap.add_argument("--threads", type=int, default=None, help="Number of threads for parallel processing (default: auto)")
    ap.add_argument("--coverage_report", action="store_true", help="Generate coverage_report.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    fill_frames(
        Path(args.frames_dir),
        Path(args.mask_path),
        Path(args.motion_csv),
        Path(args.out_dir),
        args.window,
        args.method,
        donor_side=args.donor_side, # Keyword argument
        sun_azimuth=args.sun_azimuth, # Must also be keyword
        feather=args.feather,         # Must also be keyword
        transparent=args.transparent, # Must also be keyword
        patch_smooth=args.patch_smooth, # Must also be keyword
        threads=args.threads,         # Must also be keyword
        coverage_report=args.coverage_report, # Must also be keyword
    )


if __name__ == "__main__":
    main()
