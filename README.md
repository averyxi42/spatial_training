## spatial_training

Minimal utilities for **VLM SFT + Pose supervision**.
- `training_scripts/train_sft.py`: SFT training
- `training_scripts/train_pose.py`: Pose training (adds pose loss)
- `training_scripts/build_pose_dataset.py`: build a HuggingFace dataset folder via `save_to_disk()`
- `utils/`: collators, resize transform, pose loss, feature extractor

### 1) Setup (Conda + pip)

```bash
cd /Projects/SG_VLN_HumanData/spatial_training
conda create -y -p /root/conda_envs/spatial_training python=3.11
conda activate /root/conda_envs/spatial_training
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

### 2) Train

Single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

SFT:

```bash
python3 training_scripts/train_sft.py \
  --model_id /Projects/SG_VLN_HumanData/SG-VLN/sft_pipeline/text_adapted_model \
  --train_dataset_dir /Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/train \
  --eval_dataset_dir /Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/validation \
  --output_dir ./dump/sft_training_continue
```

Pose:

```bash
python3 training_scripts/train_pose.py \
  --model_id /Projects/SG_VLN_HumanData/SG-VLN/sft_pipeline/text_adapted_model \
  --train_dataset_dir /Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/train \
  --eval_dataset_dir /Projects/SG_VLN_HumanData/spatial_training/data/habitat_web_pose_v1/validation \
  --output_dir ./dump/pose_training_continue \
  --resume_from_checkpoint /Projects/SG_VLN_HumanData/spatial_training/dump/pose_training_test/checkpoint-9500
```
