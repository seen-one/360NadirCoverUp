import os
import subprocess
import sys

inputfolder = r"H:\fullfpstest_3\jpg"
maskImagePath = r"H:\fullfpstest_3\mask.png"

parallel_workers = "8"

feather = 30

# GPS Tracking Configuration
cameraHeadingOffset = 0.0   # Degrees to add to GPS heading (e.g., 90 if camera faces right)
gpsSpeedThreshold = 2.0     # Minimum speed (m/s) to update heading; below this, hold last heading
gpxTimeOffset = -11         # Hours to add to GPX time to get true UTC (e.g., -11 if camera saved local time as UTC)
patch_smooth = 50
debugStep3 = False


# Frame step: use every nth frame for donor and final images (1 = use all frames)
frame_step = 5
use_all_frames_for_donors = False


inputHeight = 3840
inputWidth = inputHeight * 2
#planarSize = inputHeight / 2
planarSize = inputHeight /2
planarFov = 160

skipStep1 = True
skipStep2 = True
skipStep3 = False
skipStep4 = False
skipStep5 = False

OutputStep1 = os.path.join(inputfolder, "1")
OutputStep2 = os.path.join(inputfolder, "2")
OutputStep3 = os.path.join(inputfolder, "3")
OutputStep4 = os.path.join(inputfolder, "4")
OutputStep5 = os.path.join(inputfolder, "5")

os.makedirs(OutputStep1, exist_ok=True)
os.makedirs(OutputStep2, exist_ok=True)
os.makedirs(OutputStep3, exist_ok=True)
os.makedirs(OutputStep4, exist_ok=True)
os.makedirs(OutputStep5, exist_ok=True)

def Run(script, *args):
    # Use the same Python interpreter that's running this controller script
    subprocess.run([sys.executable, script] + list(args))


def RunCmd(cmd, *args):
    """Run an external command directly (not via `python`)."""
    subprocess.run([cmd] + list(args))

# Step 1 generate equirectangular to planar top down for all images in folder
if not skipStep1:
    print("\n--- Step 1/5: Converting equirectangular to planar top-down ---")
    Run("scripts/1_equirect_to_planar.py", inputfolder, OutputStep1, str(int(planarSize)), str(int(planarFov)), parallel_workers)

# Step 2 motion estimation
if not skipStep2:
    print("\n--- Step 2/5: Estimating motion (homography) ---")
    Run(
        "scripts/2_estimate_motion.py",
        "--frames_dir", OutputStep1,
        "--mask_path", maskImagePath,
        "--output_csv", os.path.join(OutputStep2, "motion.csv"),
        "--method", "homography",
        "--size", "1920"
    )

# Step 3 fill mask from neighbors
if not skipStep3:
    print("\n--- Step 3/5: Filling masked areas from neighbor frames ---")
    donor_step = 1 if use_all_frames_for_donors else frame_step
    cmd = [
        "scripts/3_fill_mask_from_neighbors.py",
        "--frames_dir", OutputStep1,
        "--mask_path", maskImagePath,
        "--motion_csv", os.path.join(OutputStep2, "motion.csv"),
        "--out_dir", OutputStep3,
        "--src_dir", inputfolder,
        "--window", "0",
        "--method", "nearest",
        "--donor_side", "auto",
        "--feather", str(feather),
        "--patch_smooth", str(patch_smooth),
        "--threads", parallel_workers,
        "--transparent",
        "--step", str(frame_step),
        "--gps_offset", str(cameraHeadingOffset),
        "--gps_threshold", str(gpsSpeedThreshold),
        "--gps_time_offset", str(gpxTimeOffset),
    ]
    if debugStep3:
        cmd.append("--debug")
    Run(*cmd)

# Step 4 convert mask to equirectangular
if not skipStep4:
    print("\n--- Step 4/5: Converting patched areas back to equirectangular ---")
    Run("scripts/4_planar_to_equirect.py", OutputStep3, OutputStep4, str(int(inputHeight)), str(int(inputWidth)), str(int(planarFov)), parallel_workers)

# Step 5 overlay patched nadir area back onto original image
if not skipStep5:
    print("\n--- Step 5/5: Overlaying patched areas onto original images ---")
    Run("scripts/5_overlay_nadir.py", inputfolder, OutputStep4, OutputStep5, "--threads", parallel_workers)

print("\nProcessing complete!")

# input("Press Enter to continue...")