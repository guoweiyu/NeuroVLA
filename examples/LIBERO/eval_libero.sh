#!/bin/bash

export LIBERO_HOME=/workspace/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}    

your_ckpt="/path/to/your/checkpoint.pth"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"

host="127.0.0.1"
base_port=10094
unnorm_key="franka"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_${folder_name}.log"

echo "============================================"
echo " LIBERO Evaluation Started "
echo " Time: ${TIMESTAMP}"
echo " Checkpoint: ${your_ckpt}"
echo " Log file: ${LOG_FILE}"
echo "============================================"


python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================"
echo " Run complete! Log saved in: ${LOG_FILE}"
echo "============================================"
