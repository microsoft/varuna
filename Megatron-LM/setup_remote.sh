machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
# machines=(172.16.5.94 172.16.5.96)

nservers=$1

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "ls t-nisar; python3 -c \"import apex; print('yo apex')\""
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem examples/pretrain_gpt2.sh varuna@${machines[i]}:/home/varuna/t-nisar/Megatron-LM/examples/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem run_varuna.py varuna@${machines[i]}:/home/varuna/t-nisar/Megatron-LM/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/arguments.py varuna@${machines[i]}:/home/varuna/t-nisar/Megatron-LM/megatron/
    # ssh -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "nohup python3 t-nisar/Varuna/vmss_scripts/listen_preemption.py > listen.out 2>listen.err &"
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/checkpointing.py varuna@${machines[i]}:/home/varuna/t-nisar/Megatron-LM/megatron/
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "rm t-nisar/Megatron-LM/varuna_gpt2_350m_32k_cont_morph_fresh*"
    # mkdir stats_cont_morph/${machines[i]}
    scp -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]}:t-nisar/Megatron-LM/eval-varuna_gpt2_350m_32k_cont_morph_fresh* ./stats_cont_morph/${machines[i]}
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo kill -9 \$(ps aux |grep listen | awk -F ' ' '{print \$2}')"
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "yes| pip3 uninstall apex; cd t-nisar/apex; yes| pip3 install -v --no-cache-dir --global-option=\"--pyprof\" --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./" >apex_install_$i 2>apex_err_$i &
    # scp -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/vmss_scripts/listen_preemption.py varuna@${machines[i]}:t-nisar/Varuna/vmss_scripts/
    # scp -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/apex_files/scaler.py varuna@${machines[i]}:t-nisar/apex/apex/amp/
    # scp -o "StrictHostKeyChecking no"  -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Megatron-base/Megatron-LM/megatron/training.py varuna@${machines[i]}:t-nisar/Megatron-base/Megatron-LM/megatron/
    i=$(($i+1))
done