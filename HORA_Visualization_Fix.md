# HORA Visualization Fix Guide

## Problem Summary
When running the HORA visualization script `scripts/vis_s1.sh`, the following issues were encountered:
1. NumPy compatibility error with deprecated `np.float` alias
2. CUDA compatibility warning with RTX 4090 GPU
3. Visualization window not appearing

## Environment Details
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU (CUDA capability sm_89)
- **Conda Environment**: hora
- **Python Version**: 3.8
- **Original PyTorch**: 1.10.1 with CUDA 10.2
- **Original NumPy**: 1.24.3

## Solutions Applied

### 1. NumPy Compatibility Fix
The error occurred because IsaacGym uses the deprecated `np.float` alias which was removed in NumPy 1.20+.

**Solution**: Downgrade NumPy to version 1.19.5
```bash
conda activate hora
conda run -n hora pip install numpy==1.19.5
```

### 2. RTX 4090 CUDA Support
PyTorch 1.10.1 with CUDA 10.2 doesn't support RTX 4090 (requires sm_89 support).

**Solution**: Upgrade PyTorch to 1.13.1 with CUDA 11.7
```bash
conda activate hora
conda run -n hora pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. Checkpoint Path Issue (if applicable)
If the checkpoint file is nested in an extra directory level, update the path in the script or move the checkpoint files to the expected location.

Expected path:
```
outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth
```

## Running the Visualization

After applying the fixes, run the visualization script:
```bash
conda activate hora
scripts/vis_s1.sh hora_v0.0.2
```

### Alternative: CPU Pipeline (Slower but works without GPU fixes)
If GPU issues persist, you can run with CPU pipeline:
```bash
conda activate hora
python train.py task=AllegroHandHora headless=False pipeline=cpu \
task.env.numEnvs=1 test=True \
task.env.object.type=simple_tennis_ball \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/hora_v0.0.2 \
checkpoint=outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth
```

## Verification
When running successfully with GPU, you should see:
```
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
```

## Troubleshooting

### Display Issues
- Check your display variable: `echo $DISPLAY`
- Try different display settings: `DISPLAY=:0` or `DISPLAY=:1`
- Run with explicit display: `DISPLAY=:0 scripts/vis_s1.sh hora_v0.0.2`

### Process Management
Kill hanging processes if needed:
```bash
# Find hanging processes
ps aux | grep -E "train.py|vis_s1" | grep -v grep

# Kill specific process IDs
kill -9 <PID>
```

## Notes
- The Gym deprecation warnings can be ignored - they don't affect functionality
- GPU pipeline is much faster than CPU pipeline for visualization
- PyTorch 1.13.1+cu117 provides good compatibility with RTX 40-series GPUs