#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
ckpt=$4

DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>

# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 \
python3 -m varuna.run_varuna --nstages 2 --batch_size 512 --chunk_size 4 --gpus_per_node 4 \
--no_morphing pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 18750 \
       --lr-decay-iters 18750 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend gloo \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 1 \
       --exit-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --use-cpu-initialization \
       --eval-iters 10 \
       --varuna --fp16


set +x
