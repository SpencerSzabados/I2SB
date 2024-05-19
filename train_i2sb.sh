
# Train I2SB on lysto64_random_crop_ddbm
python train.py \
    --n-gpu-per-node 1 --corrupt blur-uni \
    --name ct_pet_i2sb --dataset ct_pet \
    --dataset_dir /share/yaoliang/datasets/ct_pet/ \
    --data_image_size 256 --image_size 32 --data_image_channels 3 \
    --batch_size 64 --microbatch 32 --ot_ode \
    --beta_max 1.0 \
    --log_dir /u6/sszabado/checkpoints/i2sb/ct_pet/logs/ \
    --ckpt_path /u6/sszabado/checkpoints/i2sb/ct_pet/ \
    --ckpt 10000.pt \
    --autoencoder_ckpt /share/yaoliang/EQUIV_DDBM/AUTOENC/pet_ct/checkpoint_700000.pth/ 