export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
node_rank=$1
name=clip_pixel0.5_mid6
all_dir=/nlp_group/wuxing/suzhenpeng/mae_resnet/clip_output_dir/${name}
mkdir ${all_dir}
nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.150.143  --node_rank=${node_rank}  --nproc_per_node=8   --master_port 3243  \
    --use_env main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 300 \
    --accum_iter 1 \
    --output_dir ${all_dir} \
    --log_dir ${all_dir} \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --norm_pix_loss \
    --data_path /nlp_group/wuxing/suzhenpeng/mae_resnet/imagenet \
    >${all_dir}/${name}_${node_rank}.log 2>&1 &
    # --start_epoch 41 \
    # --resume output_dir/resnet_pretrain_100_beta1_resdd2/checkpoint-40.pth \