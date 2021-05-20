#! /bin/bash
#export PATH="/home/varuna/anaconda3/bin:$PATH"
#which conda

ckpt=$1
GPUS_PER_SERVER=1

user="rahul"

DATA_PATH=/home/$user/gpt2-blob/turing/megatron
CHECKPOINT_PATH=/home/$user/gpt2-blob/dummy

python -m varuna.run_varuna --nstages 4 --chunk_size 2 --batch_size 256 \
       --code_dir /home/rahul/Varuna/Megatron-LM \
       --gpus_per_node $GPUS_PER_SERVER --no_morphing pretrain_gpt2.py \
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
       --lr 0.0048 \
       --min-lr 1e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .05 \
       --log-interval 1 \
       --save-interval 300 \
       --max-num-ckpts 10 \
       --min-ckpt-iter-to-remove 9100 \
       --load-iteration $ckpt \
       --eval-interval 100 \
       --eval-iters 10 \
       --loss-file dummy \
       --fp16 --varuna 

# num params = 20,753,384,400
set +x
