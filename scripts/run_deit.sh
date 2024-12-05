CUDA_VISIBLE_DEVICES='1,2' python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model soft_moe_vit_tiny --batch-size 256 \
 --data-path /home/ubuntu/workspace/dataset/imagenet1K --output_dir /home/ubuntu/workspace/deit/result \
> >(tee -a /home/ubuntu/workspace/deit/result/softmoe_tiny.txt) 2>&1