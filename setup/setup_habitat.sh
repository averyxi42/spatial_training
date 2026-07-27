# We require python>=3.9 and cmake>=3.14
# conda create -n vln -y python=3.10.16 cmake=3.14.0
conda create -n vln -y python=3.12 cmake=3.27 -c conda-forge
conda activate vln
conda install habitat-sim==0.3.4 withbullet headless -c conda-forge -c aihabitat 
# pip install --no-deps -e .
# pip install ray[default]==2.53.0