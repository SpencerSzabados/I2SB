
# Train I2SB on lysto64_random_crop_ddbm
mpiexec --allow-run-as-root -n 1 python train.py \
    --n-gpu-per-node 1 --corrupt blur-uni \
    --name lysto64_i2sb --dataset lysto64 \
    --dataset_dir /share/yaoliang/datasets/lysto64_random_crop_pix2pix/AB/ \
    --data_image_size 64 --image_size 64 \
    --batch_size 128 --microbatch 32 --ot-ode \
    --beta-max 1.0 \
    --log_dir /u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/logs/