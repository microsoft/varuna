#! /bin/bash
#export PATH="/home/varuna/anaconda3/bin:$PATH"
#which conda

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
ckpt=$4
GPUS_PER_SERVER=1

echo $NNODES $NODE_RANK $MASTER_ADDR $ckpt
date
ifconfig eth0 | awk '/inet / {gsub("addr:", "", $2); print $2}'
user="rahul"

DATA_PATH=/home/$user/gpt2-blob/turing/megatron
CHECKPOINT_PATH=/home/$user/gpt2-blob/mega_2_5b_4.8e-3_clipgrads

export PATH="/home/$user/anaconda3/bin:$PATH"
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 \
python run_varuna.py --batch_size 8192 --total_num_stages 50 \
       --ngpus_per_server $GPUS_PER_SERVER --nservers $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR pretrain_gpt2.py \
       --num-layers 54 \
       --hidden-size 1920 \
       --num-attention-heads 20 \
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
       --save-interval 900 \
       --max-num-ckpts 10 \
       --min-ckpt-iter-to-remove 9100 \
       --load-iteration $ckpt \
       --eval-interval 100 \
       --eval-iters 10 \
       --loss-file varuna_2_5b_8192_lr_0.0048_clipgrads \
       --fp16 --varuna 

blobmounted=$(ls /home/$user/gpt2-blob/turing-dry-run-input)

if [ $? != 0 ]
then
    sudo blobfuse /home/rahul/gpt2-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/rahul/fuse_connection2.cfg -o allow_other
fi
# num params = 20,753,384,400
set +x
