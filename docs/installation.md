# FreelyMovingEphys -- installation
Setting up the repository for analysis.

1. Be sure that GPU drivers, CUDA, etc. are set up on the device.
2. Install anaconda., etc.
3. Clone the DeepLabCut repository locally, which can be found [on github](https://github.com/DeepLabCut/DeepLabCut).
4. Create the GPU version of the DeepLabCut anaconda environment, instructions [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md). Be sure to install it with `tensorflow-gpu`.
5. Clone the `parallax` branch of Autopilot locally, which can be found [on github](https://github.com/wehr-lab/autopilot/tree/parallax).
6. Install Autopilot into the DLC environment with `pip install -e ./autopilot/` from the `/GitHub/` directory (if installing on Windows, you will need to create the directory `Users/[username]/autopilot/logs/` by hand prior to installing the repository into the DLC environment)
7. Run `pip install -r requirements.txt` in the directory of the FreelyMovingEphys repository to install any other dependencies.
8. Also install FFMPEG, either with `sudo apt install ffmpeg` for linux or according to [these](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows) instructions for Windows installation.