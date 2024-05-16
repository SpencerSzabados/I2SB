
# Train I2SB on lysto64_random_crop_ddbm
mpiexec --allow-run-as-root -n 1 python train.py \
    --n-gpu-per-node 1 --corrupt blur-uni \
    --name lysto64_i2sb --dataset lysto64 \
    --dataset_dir /home/sszabados/datasets/lysto64_random_crop_pix2pix/AB/ \
    --data_image_size 64 --image_size 64 \
    --batch_size 32 --microbatch 32 --ot-ode \
    --beta-max 1.0 \
    --log_dir /home/sszabados/checkpoints/i2sb/lysto64_random_crop/logs/ \
    --ckpt_path /home/sszabados/checkpoints/i2sb/lysto64_random_crop/