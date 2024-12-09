CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.launch --master_port=10011 --nproc_per_node=2 --use_env main3.py --model soft_moe_vit_tiny --batch-size 128 \
 --data-path /home/ubuntu/workspace/dataset/imagenet1K --output_dir /home/ubuntu/workspace/deit/result/softmoe_fbell \
#  > >(tee -a /home/ubuntu/workspace/deit/result/softmoe_Tfbell.txt) 2>&1