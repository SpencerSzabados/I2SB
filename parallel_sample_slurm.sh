#!/bin/bash

#SBATCH --job-name=parallel_fid_sample_I2SB
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --array=12-22
#SBATCH --time=3:00:00
#SBATCH --signal=USR1
#SBATCH --signal=B:USR1@60
#SBATCH --export=ALL
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log


export PYTHONPATH="${PYTHONPATH}:/ssd005/projects/watml/shared_conda"
export OMPI_MCA_opal_cuda_support=true

handler() 
{
    echo "###############################"    && 
    echo "function handler called at $(date)" &&
    echo "###############################"    && 

    sbatch ${BASH_SOURCE[0]} 
}

trap handler SIGUSR1

# Define the index for this task (0-49)
TASK_INDEX=${SLURM_ARRAY_TASK_ID}

# Print some information to the output file
echo "Starting task with index ${TASK_INDEX}"

# Run the Python script with the task index
/ssd005/projects/watml/shared_conda/i2sb/bin/python \
    /ssd005/projects/watml/szabados/models/I2SB/image_sample.py \
        --n-gpu-per-node 1 --ot-ode \
        --name lysto64_i2sb --dataset lysto64 \
        --dataset_dir /ssd005/projects/watml/data/lysto64_random_crop_pix2pix/AB/ \
        --data_image_size 64 --image_size 64 \
        --num_samples 1000 \
        --partition ${TASK_INDEX} \
        --batch_size 128 \
        --log_dir /ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/ \
        --ckpt_path /ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/ \
        --ckpt 110000.pt \
        --save_dir /ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/fid_samples/

# Print completion message
echo "Task with index ${TASK_INDEX} completed"

wait