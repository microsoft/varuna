#! /bin/bash


DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>

# 355m model
$NUM_LAYERS=24
$HIDDEN_SIZE=1024
$NUM_ATTENTION_HEADS=16

# # 1.5bn model
# $NUM_LAYERS=48
# $HIDDEN_SIZE=1600
# $NUM_ATTENTION_HEADS=16

# # 2.5bn model
# $NUM_LAYERS=54
# $HIDDEN_SIZE=1920
# $NUM_ATTENTION_HEADS=20

# #8.3bn model
# $NUM_LAYERS=72
# $HIDDEN_SIZE=3072
# $NUM_ATTENTION_HEADS=32


# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 \
python3 -m varuna.run_varuna --nstages 2 --batch_size 8192 --chunk_size 4 --gpus_per_node 4 \
--no_morphing pretrain_gpt2.py \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
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
