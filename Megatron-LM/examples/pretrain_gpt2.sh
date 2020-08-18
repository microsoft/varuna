#! /bin/bash


RANK=0
WORLD_SIZE=2

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
ckpt=$4

date
echo $NNODES $NODE_RANK $MASTER_ADDR $ckpt

DATA_PATH=/home/varuna/bert-large-blob/openwebtext_text_document
CHECKPOINT_PATH=/home/varuna/bert-large-blob/varuna_gpt2_350m_32k_8x32_cont_morph/

python3 run_varuna.py --nstages 8 --batch_size 32768 --chunk_size 10 --total_num_stages 24 --ngpus_per_server 4 --nservers $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR  pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 7813 \
       --lr-decay-iters 5000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend gloo \
       --lr 0.0012 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .2 \
       --log-interval 10 \
       --save-interval 15 \
       --max-num-ckpts 3 \
       --min-ckpt-iter-to-remove 3400 \
       --eval-interval 300 \
       --eval-iters 10 \
       --loss-file varuna_gpt2_350m_32k_cont_morph_fresh \
       --fp16 --varuna \
       --load-iteration $ckpt
set +x
