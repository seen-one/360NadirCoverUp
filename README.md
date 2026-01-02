
# NadirCoverUp

Reconstruction of the nadir area of 360 photos using homography to hide the car where the 360 camera is attached to like in Google Street View.

The direction of the sun is automatically calculated using the time and GPS information to choose the side facing the sun to avoid shadows cast by the car.

Vibe coded initially using GitHub Copilot, later with Google Antigravity.

## Install

`git clone https://github.com/seen-one/NadirCoverUp.git`
`cd NadirCoverUp`
`python -m venv env`
`.\env\Scripts\activate` (Windows)
`source env/bin/activate` (Linux/Mac)
`pip install -r requirements.txt`
`exit`

On Windows, exiftool.exe should be installed in the same directory as the script.

## Usage

Your 360 camera should record at a high enough framerate (15-30 fps) for sufficient tracking.

Configure your inputfolder and maskImagePath in `NadirCoverUp.py`.
inputfolder should contain an image sequence with output `Output_0001.jpg`, `Output_0002.jpg`, etc.
The .gpx file should be placed in the same folder as inputfolder.
If desired, you may edit the other parameters to suit your needs.

`.\env\Scripts\activate` (Windows)
`source env/bin/activate` (Linux/Mac)
`python NadirCoverUp.py`

The script will create a new folder at `OutputStep6` with the processed images, ready for upload to Mapillary or Panoramax. If wanting to upload to Google Street View, you can use ffmpeg to encode a video from `OutputStep5` image sequence.