#!/bin/bash
mkdir -p checkpoints

# Training
python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 3 4 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001


# NIPQ Training
python -u train.py --name raft-things-nipq --stage things --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --quantize --ft_epoch 90000 --target 8 --last_fp

# KETI Pruning
python -u train.py --name raft-things-p0.1 --stage things --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --pruning --prune_ratio 0.1

# KETI Pruning + NIPQ
python -u train.py --name raft-things-p0.1-nipq --stage things --validation sintel --restore_ckpt checkpoints/raft-things-p0.1.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --quantize --ft_epoch 90000 --target 8 --pruning  --prune_ratio 0.1