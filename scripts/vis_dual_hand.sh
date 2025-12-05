#!/bin/bash
# Visualization script for dual hand (Allegro + LEAP) demo
# Usage: ./scripts/vis_dual_hand.sh <checkpoint_folder>
# Example: ./scripts/vis_dual_hand.sh hora_s1

CACHE=$1

if [ -z "$CACHE" ]; then
    echo "Usage: $0 <checkpoint_folder>"
    echo "Example: $0 hora_s1"
    exit 1
fi

python train.py task=AllegroHandHoraWithLeap headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=simple_tennis_ball \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth
