# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
machines=($(cat /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out))

nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo rm t-nisar/Varuna -r; python3 -c \"import apex; print('yo apex')\""
    #scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/varuna/varuna.py varuna@${machines[i]}:/home/varuna/t-nisar/Varuna/varuna.py
    #  scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem examples/pretrain_gpt2.sh varuna@${machines[i]}:/home/varuna/t-nisar/Varuna/Megatron-LM/examples/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem run_varuna.py varuna@${machines[i]}:/home/varuna/t-nisar/Varuna/Megatron-LM/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/training.py varuna@${machines[i]}:/home/varuna/t-nisar/Varuna/Megatron-LM/megatron/
    # ssh -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "nohup python3 t-nisar/Varuna/vmss_scripts/listen_preemption.py > listen.out 2>listen.err &"
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/checkpointing.py varuna@${machines[i]}:/home/varuna/t-nisar/Varuna/Megatron-LM/megatron/
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "cd t-nisar/Varuna; sudo python3 setup.py develop" & 
    # mkdir -p gantt_logs/${machines[i]}
    #scp -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]}:t-nisar/Varuna/Megatron-LM/varuna_logs* ./gantt_dynamic_schedule/
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo kill -9 \$(ps aux |grep listen | awk -F ' ' '{print \$2}')"
    ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "ps aux | grep listen"
    # scp -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/vmss_scripts/listen_preemption.py varuna@${machines[i]}:t-nisar/Varuna/vmss_scripts/
    # scp -i /home/varuna/.ssh/vdummy.pem ../apex_files/handle.py varuna@${machines[i]}:apex/apex/amp/
    # scp -i /home/varuna/.ssh/vdummy.pem ../apex_files/scaler.py varuna@${machines[i]}:apex/apex/amp/
    # scp -i /home/varuna/.ssh/vdummy.pem ../varuna/__init__.py varuna@${machines[i]}:t-nisar/Varuna/varuna/
    # scp -i /home/varuna/.ssh/vdummy.pem ../varuna/partitioned_model.py varuna@${machines[i]}:t-nisar/Varuna/varuna/
    # scp -o "StrictHostKeyChecking no"  -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Megatron-base/Megatron-LM/megatron/training.py varuna@${machines[i]}:t-nisar/Megatron-base/Megatron-LM/megatron/
    i=$(($i+1))
done
