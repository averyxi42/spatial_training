#!/bin/bash
# Builds the third conda env `mshab` (SAPIEN/ManiSkill-HAB sim actors).
# Lives on /Projects/envs because the container's root overlay is ~99% full.
# Ray reaches it by name via a symlink under $CONDA_DIR/envs.
set -euo pipefail
PREFIX=/Projects/envs/mshab
REPO=/Projects/spatial_training_mshab
TP=$REPO/third_party
CONDA=/workspace/conda/bin/conda

$CONDA create -y --prefix $PREFIX python=3.10.16
PIP="$PREFIX/bin/pip"
# Same torch/ray as the VLM env so Ray object pickles and cluster versions match.
$PIP install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
$PIP install "ray[default]==2.53.0" "numpy==1.26.4"
[ -d $TP/ManiSkill ] || git clone --depth 1 -b mshab --single-branch https://github.com/haosulab/ManiSkill.git $TP/ManiSkill
[ -d $TP/mshab ]     || git clone --depth 1 https://github.com/arth-shukla/mshab.git $TP/mshab
$PIP install -e $TP/ManiSkill
$PIP install -e $TP/mshab
# LongNav itself (core deps only: hydra, regex; ray already pinned above)
$PIP install --no-dependencies -e $REPO
$PIP install hydra-core regex
ln -sfn $PREFIX /workspace/conda/envs/mshab
$PREFIX/bin/python - <<'PY'
import numpy, torch, ray, sapien, mani_skill, mshab, gymnasium
print("numpy", numpy.__version__, "torch", torch.__version__, "ray", ray.__version__,
      "sapien", sapien.__version__, "mani_skill", mani_skill.__version__, "gymnasium", gymnasium.__version__)
PY
echo INSTALL_OK
