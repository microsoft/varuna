# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
machines=($(cat /home/varuna/t-saathl/tieweights/Varuna/Megatron-LM/available_machines.out))
# machines=(10.0.3.5)
nservers=${#machines[@]}

i=0 
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}

    # ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "sudo umount /home/varuna/gpt2-blob"
    ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "sudo blobfuse /home/varuna/gpt2-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/varuna/fuse_connection2.cfg -o allow_other"
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "sudo rm -r /home/varuna/t-saathl/Varuna "
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem -r /home/varuna/t-saathl/tieweights/Varuna varuna@${machines[i]}:/home/varuna/t-saathl/. &

    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem -r /home/varuna/t-saathl/tieweights/Varuna/Megatron-LM/examples/pretrain_gpt2.sh varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/examples/.
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-saathl/tieweights/Varuna/varuna/varuna.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/varuna/varuna.py

    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem examples/pretrain_gpt2.sh varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/examples/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem run_varuna.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/ 
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/model/gpt2_model.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/model 
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/data/gpt2_dataset.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/data 
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/initialize.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/
    #scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/training.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/ 
    # scp -i /home/varuna/.ssh/vdummy.pem ../varuna/varuna.py varuna@${machines[i]}:t-saathl/Varuna/varuna/ 
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "sudo blobfuse /home/varuna/bert-large-blob --log-level=LOG_DEBUG --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/varuna/fuse_connection.cfg -o allow_other" 
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "cat ~/local_ckpt_tracker.txt "
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "sudo blobfuse /home/varuna/bert-large-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/varuna/fuse_connection.cfg -o allow_other" &
    # mkdir -p gantt_logs/${machines[i]}
    #scp -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]}:t-nisar/Varuna/Megatron-LM/varuna_logs* ./gantt_dynamic_schedule/
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "ps aux | grep listen"
    # scp -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/vmss_scripts/listen_preemption.py varuna@${machines[i]}:t-nisar/Varuna/vmss_scripts/
    # scp -i /home/varuna/.ssh/vdummy.pem ../varuna/partitioned_model.py varuna@${machines[i]}:t-saathl/Varuna/varuna/ &
    i=$(($i+1))
done
