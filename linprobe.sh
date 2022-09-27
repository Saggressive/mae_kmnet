export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

node_rank=$1
name=lin/clip_pixel0.5_mid6
all_dir=/nlp_group/wuxing/suzhenpeng/mae_resnet/clip_output_dir/${name}
mkdir ${all_dir}


nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.146.14  --node_rank=${node_rank}  --nproc_per_node=8   --master_port 23332  \
    --use_env main_linprobe.py  \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune clip_output_dir/clip_pixel0.5_mid6/checkpoint-299.pth \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --data_path /share/a100/vig_pytorch/imagenet-2012 \
    --output_dir ${all_dir} \
    --log_dir ${all_dir} \
    >${all_dir}/${node_rank}.log 2>&1 &