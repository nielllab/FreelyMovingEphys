# FreelyMovingEphys -- installation
Setting up the repository for analysis.

1. Install FFmpeg, either with `sudo apt install ffmpeg` for or according to [these](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows) instructions for Windows. It should be installed in `C:\Program Files\ffmpeg\` in Windows.
2. Install anaconda [here](https://www.anaconda.com/products/individual).
3. Install GPU drivers [here](https://www.nvidia.com/Download/index.aspx).
4. Install CUDA.
5. Clone the DeepLabCut repository locally, which can be found [on github](https://github.com/DeepLabCut/DeepLabCut).
6. Create the GPU version of the DeepLabCut anaconda environment, instructions [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md).
7. Run `conda install -c conda-forge cudnn` in the conda environment.
8. Run `pip install -r requirements.txt` in the directory of the FreelyMovingEphys repository to install any other dependencies.