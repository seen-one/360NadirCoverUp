import os
import subprocess
import sys
import glob

inputfolder = r"H:\fullfpstest_4\jpg"
maskImagePath = r"H:\fullfpstest_4\mask.png"

parallel_workers = "8"

feather = 20

# GPS Tracking Configuration
cameraHeadingOffset = 0.0   # Degrees to add to GPS heading (e.g., 90 if camera faces right)
gpsSpeedThreshold = 2.0     # Minimum speed (m/s) to update heading; below this, hold last heading
gpxTimezoneOffset = -10         # Hours offset for Qoocam incorrectly reporting local time as UTC (e.g. -10 for AEST)
gpxCaptureOffset = 2
patch_smooth = 20
debugStep3 = False

# Step 6 Specific Configuration
gpxPath = ""                # Path to GPX file. If empty, will look for the first .gpx in inputfolder.
video_fps = 24               # FPS of the original timelapse video
min_distance = 5            # Minimum distance (m) between pictures

limit_frames = 0  # Process only the first n frames; 0 or for all (before frame stepping)

# Frame step: use every nth frame for donor and final images (1 = use all frames)
frame_step = 4
use_all_frames_for_donors = False


inputHeight = 3840
inputWidth = inputHeight * 2
planarSize = inputHeight /2
planarFov = 160

skipStep1 = True
skipStep2 = True
skipStep3 = True
skipStep4 = True
skipStep5 = True
skipStep6 = False

OutputStep1 = os.path.join(inputfolder, "1")
OutputStep2 = os.path.join(inputfolder, "2")
OutputStep3 = os.path.join(inputfolder, "3")
OutputStep4 = os.path.join(inputfolder, "4")
OutputStep5 = os.path.join(inputfolder, "5")
OutputStep6 = os.path.join(inputfolder, "6")

os.makedirs(OutputStep1, exist_ok=True)
os.makedirs(OutputStep2, exist_ok=True)
os.makedirs(OutputStep3, exist_ok=True)
os.makedirs(OutputStep4, exist_ok=True)
os.makedirs(OutputStep5, exist_ok=True)
os.makedirs(OutputStep6, exist_ok=True)

def Run(script, *args):
    # Use the same Python interpreter that's running this controller script
    subprocess.run([sys.executable, script] + list(args))


def RunCmd(cmd, *args):
    """Run an external command directly (not via `python`)."""
    subprocess.run([cmd] + list(args))

# Step 1 generate equirectangular to planar top down for all images in folder
if not skipStep1:
    print("\n--- Step 1/6: Converting equirectangular to planar top-down ---")
    cmd = ["scripts/1_equirect_to_planar.py", inputfolder, OutputStep1, str(int(planarSize)), str(int(planarFov)), parallel_workers]
    if limit_frames: cmd += ["--limit", str(limit_frames)]
    Run(*cmd)

# Step 2 motion estimation
if not skipStep2:
    print("\n--- Step 2/6: Estimating motion (homography) ---")
    cmd = [
        "scripts/2_estimate_motion.py",
        "--frames_dir", OutputStep1,
        "--mask_path", maskImagePath,
        "--output_csv", os.path.join(OutputStep2, "motion.csv"),
        "--method", "homography",
        "--size", str(int(planarSize))
    ]
    if limit_frames: cmd += ["--limit", str(limit_frames)]
    Run(*cmd)

# Step 3 fill mask from neighbors
if not skipStep3:
    print("\n--- Step 3/6: Filling masked areas from neighbor frames ---")
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
        "--gps_time_offset", str(gpxTimezoneOffset),
    ]
    if debugStep3:
        cmd.append("--debug")
    if limit_frames: cmd += ["--limit", str(limit_frames)]
    Run(*cmd)

# Step 4 convert mask to equirectangular
if not skipStep4:
    print("\n--- Step 4/6: Converting patched areas back to equirectangular ---")
    cmd = ["scripts/4_planar_to_equirect.py", OutputStep3, OutputStep4, str(int(inputHeight)), str(int(inputWidth)), str(int(planarFov)), parallel_workers]
    if limit_frames: cmd += ["--limit", str(limit_frames)]
    Run(*cmd)

# Step 5 overlay patched nadir area back onto original image
if not skipStep5:
    print("\n--- Step 5/6: Overlaying patched areas onto original images ---")
    cmd = ["scripts/5_overlay_nadir.py", inputfolder, OutputStep4, OutputStep5, "--threads", parallel_workers]
    if limit_frames: cmd += ["--limit", str(limit_frames)]
    Run(*cmd)

# Step 6 GPS Tagging and EXIF metadata
if not skipStep6:
    print("\n--- Step 6/6: Applying GPS tags and EXIF metadata ---")
    
    current_gpx = gpxPath
    if not current_gpx:
        gpx_files = glob.glob(os.path.join(inputfolder, "*.gpx"))
        if gpx_files:
            current_gpx = gpx_files[0]
            print(f"Auto-detected GPX: {current_gpx}")
        else:
            print("Warning: No GPX file found. Skipping Step 6.")
            current_gpx = None
            
    if current_gpx:
        cmd = [
            "scripts/qoocam2panoramax.py",
            "--input", OutputStep5,
            "--output", OutputStep6,
            "--gpx", current_gpx,
            "--fps", str(video_fps),
            "--heading", str(int(cameraHeadingOffset)),
            "--offset", str(gpxCaptureOffset),
            "--distance", str(min_distance),
            "--timezone", str(gpxTimezoneOffset)
        ]
        if limit_frames: cmd += ["--limit", str(limit_frames)]
        Run(*cmd)

print("\nProcessing complete!")

# input("Press Enter to continue...")
