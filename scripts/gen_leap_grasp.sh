#!/bin/bash
# Generate grasp cache for LEAP hand
# Usage: ./scripts/gen_leap_grasp.sh <GPU_ID> <SCALE>
# Example: ./scripts/gen_leap_grasp.sh 0 0.8
#
# IMPORTANT:
# - pipeline=cpu is REQUIRED for contact detection
# - numEnvs=20000 for faster grasp collection
# - episodeLength=50 short episodes for sampling
# - Outputs to: cache/leap_leap_internal_grasp_50k_s<SCALE>.npy

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hora

GPUS=$1
SCALE=$2

if [ -z "$GPUS" ] || [ -z "$SCALE" ]; then
    echo "Usage: ./scripts/gen_leap_grasp.sh <GPU_ID> <SCALE>"
    echo "Example: ./scripts/gen_leap_grasp.sh 0 0.8"
    exit 1
fi

echo "Generating LEAP hand grasp cache..."
echo "GPU: ${GPUS}, Scale: ${SCALE}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=LeapHandGrasp headless=True pipeline=cpu sim_device=cpu \
task.env.numEnvs=20000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=simple_tennis_ball \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True
