
BERT SQUAD pipeliing

To run Varuna for BERT use the script launch_bert.py and it's worker script varuna_worker.py with desired parameters

Example:

GLOO_SOCKET_IFNAME=eth0  python launch_bert.py --node_rank 1 --nservers 2 --ngpus_per_server 4 --master_addr 10.4.0.29 varuna_worker.py --model_name_or_path bert-large-cased --do_train --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json  --learning_rate 3e-5 --num_train_epochs 2.0 --logging_steps 20 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir debug_squad --chunks=42 --per_gpu_train_batch_size=1260