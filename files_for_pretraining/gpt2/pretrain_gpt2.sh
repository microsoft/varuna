#! /bin/bash


RANK=0
WORLD_SIZE=2

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3

echo $NNODES $NODE_RANK $MASTER_ADDR

DATA_PATH=/home/varuna/bert-large-blob/openwebtext-subset_text_document
CHECKPOINT_PATH=/home/varuna/bert-large-blob/megaperf

python3 run_varuna.py --nstages 96 --batch_size 8096 --chunk_size 2 --total_num_stages 96 --ngpus_per_server 4 --nservers $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR  pretrain_gpt2.py \
       --num-layers 96 \
       --hidden-size 4200 \
       --num-attention-heads 42 \
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
       --log-interval 1 \
       --save-interval 15 \
       --max-num-ckpts 3 \
       --eval-interval 10 \
       --eval-iters 10 \
       --loss-file testing \
       --fp16 --varuna
# num params = 20,753,384,400
set +x
