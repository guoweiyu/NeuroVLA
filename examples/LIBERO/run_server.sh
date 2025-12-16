#!/bin/bash

your_ckpt=/path/to/your/libero_checkpoint.pth
base_port=10093
# export DEBUG=true

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16