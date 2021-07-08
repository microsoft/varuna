
user="gandiva"

DATA_PATH=/home/$user/blobcontainer/openwebtext-subset_text_document
GPUS_PER_SERVER=4

NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO \
python -m varuna.run_varuna --nstages 1 --chunk_size 1 --batch_size 256 \
        --gpus_per_node $GPUS_PER_SERVER --no_morphing pretrain_gpt2.py \
        --num-layers 54 \
        --hidden-size 1920 \
        --num-attention-heads 20 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --train-iters 100 \
        --lr-decay-iters 100 \
        --data-path $DATA_PATH \
        --distributed-backend gloo \
        --vocab-file gpt2-vocab.json \
        --merge-file gpt2-merges.txt \
        --save /home/$user/blobcontainer/2_5b_profile \
        --save-interval 1000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00001 \
        --min-lr 1e-5 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --use-cpu-initialization \
        --warmup .05 \
        --fp16 \
        --varuna \
        --profiling

