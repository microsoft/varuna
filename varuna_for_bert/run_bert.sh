# usage: ./varuna_cmd.sh GPU rank partitions chunks batch_size
# set $SQUAD_DIR to directory holding train and predict files

CUDA_VISIBLE_DEVICES=$1 GLOO_SOCKET_IFNAME=eth0  python varuna_worker.py --model_name_or_path bert-large-cased --do_train --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --per_gpu_train_batch_size=$5  --learning_rate 3e-5 --num_train_epochs 2.0 --logging_steps 20 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir debug_squad --chunks=$4 --partitions=$3 --rank=$2

# ls | grep $1