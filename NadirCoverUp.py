import os
import subprocess
import sys

inputfolder = r"C:\Users\m\Documents\GitHub\NadirCoverUp\sample\3fps"
maskImagePath = r"C:\Users\m\Documents\GitHub\NadirCoverUp\sample\3840a.png"

cpuThreads = "4"

initialSunAngle = 180.0
feather = 20
patch_smooth = 50

inputHeight = 3840
inputWidth = inputHeight * 2
#planarSize = inputHeight / 2
planarSize = inputHeight /4
planarFov = 160

skipStep1 = True
skipStep2 = True
skipStep3 = False
skipStep4 = True
skipStep5 = True
skipStep6 = True

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
    Run("scripts/1_equirect_to_planar.py", inputfolder, OutputStep1, str(int(planarSize)), str(int(planarFov)), cpuThreads)

# Step 2 motion estimation
if not skipStep2:
    Run(
        "scripts/2_estimate_motion.py",
        "--frames_dir", OutputStep1,
        "--mask_path", maskImagePath,
        "--output_csv", os.path.join(OutputStep2, "motion.csv"),
        "--method", "homography",
        "--size", "512"
    )

# Step 3 fill mask from neighbors
if not skipStep3:
    Run(
        "scripts/3_fill_mask_from_neighbors.py",
        "--frames_dir", OutputStep1,
        "--mask_path", maskImagePath,
        "--motion_csv", os.path.join(OutputStep2, "motion.csv"),
        "--out_dir", OutputStep3,
        "--window", "0",
        "--method", "nearest",
        "--donor_side", "auto",
        "--sun_azimuth", str(int(initialSunAngle)),
        "--feather", str(feather),
        "--patch_smooth", str(patch_smooth),
        "--threads", cpuThreads,
        "--transparent",
    )

# Step 4 convert mask to equirectangular
if not skipStep4:
    Run("scripts/4_planar_to_equirect.py", OutputStep3, OutputStep4, str(int(inputHeight)), str(int(inputWidth)), str(int(planarFov)), cpuThreads)

# Step 5 overlay patched nadir area back onto original image
if not skipStep5:
    Run("scripts/5_overlay_nadir.py", inputfolder, OutputStep4, OutputStep5, "--threads", cpuThreads)

# Step 6 sharpen result - see if there is a way to do it not using imagemagick
if not skipStep6:
    Run("scripts/6_sharpen.py", OutputStep4, OutputStep5)

# input("Press Enter to continue...")