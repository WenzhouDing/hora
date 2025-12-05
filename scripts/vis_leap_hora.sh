#!/bin/bash
# Visualization script for LEAP Hand Hora: Running Allegro policy on LEAP hand
# LEAP is primary (manipulates object), Virtual Allegro shows what policy sees
#
# Usage: ./scripts/vis_leap_hora.sh <checkpoint_folder>
# Example: ./scripts/vis_leap_hora.sh hora_v0.0.2
#
# This expects a checkpoint trained with AllegroHandHora at:
#   outputs/AllegroHandHora/<checkpoint_folder>/stage1_nn/best.pth

CACHE=$1

if [ -z "$CACHE" ]; then
    echo "Usage: $0 <checkpoint_folder>"
    echo "Example: $0 hora_v0.0.2"
    echo ""
    echo "This runs a trained Allegro Hora policy on LEAP hand."
    echo "The LEAP hand (right) is the primary hand manipulating the object."
    echo "The Virtual Allegro hand (left) shows what the policy sees/outputs."
    exit 1
fi

python train.py task=LeapHandHora headless=False pipeline=gpu \
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
