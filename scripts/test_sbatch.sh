#!/bin/bash -login

VENV=$HOME/src/generic_pose/bpy
source $VENV/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1}

echo "Using GPU: "${1}

cd ../scripts
python speedtest.py ${@:2}

deactivate
exit
