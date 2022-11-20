#   Commands initially used to isolate python + dependencies while developing.

#   Create a basic environment with just python
conda create -y -p $PWD/chess_env python=3.8

#   Add packages we'll need
conda activate $PWD/chess_env
python -m pip install numpy pandas tensorflow scipy
conda deactivate
