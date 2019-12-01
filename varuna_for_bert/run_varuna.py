finalcmd = ''
partitions = 4
chunks = 42
batch_size = 630

for rank in range(4):
    # cmd = 'CUDA_VISIBLE_DEVICES='+str(rank)+' GLOO_SOCKET_IFNAME=eth0  python run_squad.py --model_type bert --model_name_or_path bert-large-cased --do_train --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --per_gpu_train_batch_size=360  --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir debug_squad --chunks=12 --partitions=4 --rank='+str(rank)+' & '
    cmd = 'bash run_bert.sh '+str(rank)+' '+str(rank)+' '+str(partitions)+' '+str(chunks)+' '+str(batch_size)+' & '
    finalcmd += cmd

# for rank in range(4,8):
#     cmd = ' ssh p40-gpu-0004 source activate torch_exp; cd ~/dist_tf/torch/paramodel/pytorch-transformers/examples/; CUDA_VISIBLE_DEVICES='+str(rank-4)+' GLOO_SOCKET_IFNAME=eth0  python run_squad.py --model_type bert --model_name_or_path bert-large-cased --do_train --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --per_gpu_train_batch_size=360  --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir debug_squad --chunks=12 --partitions=8 --rank='+str(rank)+' & '
#     finalcmd += cmd

print(finalcmd)

import os
os.system(finalcmd)