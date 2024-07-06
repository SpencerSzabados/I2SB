#!/bin/bash

#SBATCH --job-name=compute_fid_I2SB
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

# Print some information to the output file
echo "Starting task..."

# Run the Python script with the task index
/ssd005/projects/watml/shared_conda/i2sb/bin/python \
    /ssd005/projects/watml/szabados/models/I2SB/train.py \
        --n-gpu-per-node 1 --ot-ode \
        --name lysto64_i2sb --dataset lysto64 \
        --dataset_dir /ssd005/projects/watml/data/lysto64_random_crop_pix2pix/AB/ \
        --data_image_size 64 --image_size 64 \
        --batch_size 128 \
        --log_dir /ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/ \
        --ckpt_path /ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/ \
        --ckpt 110000.pt 
        
# Print completion message
echo "Task completed."

wait