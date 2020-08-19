machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))

nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "ls t-nisar; python3 -c \"import apex; print('yo apex')\""
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem examples/pretrain_gpt2.sh varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/examples/
    scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem pretrain_gpt2.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/
    # scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/model/language_model.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/model
    # ssh -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "rm t-saathl/Varuna/Megatron-LM/_tmp*"
    scp  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem megatron/training.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/Megatron-LM/megatron/
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "export PATH=\"/home/varuna/anaconda3/bin:\$PATH\"; yes| pip uninstall apex; cd apex; /home/varuna/anaconda3/bin/python -m  pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./" > apex_out_$i 2>apex_err_$i &
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "export PATH=\"/home/varuna/anaconda3/bin:\$PATH\"; cd /home/varuna/t-saathl/Varuna/varuna; g++ generate_schedule.cc -o genschedule" 
    # ssh  -o "StrictHostKeyChecking no" -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "export PATH=\"/home/varuna/anaconda3/bin:\$PATH\"; cd /home/varuna/t-saathl/Varuna; /home/varuna/anaconda3/bin/python setup.py develop"    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo kill -9 \$(ps aux |grep listen | awk -F ' ' '{print \$2}')"
    # scp -i /home/varuna/.ssh/vdummy.pem ../varuna/varuna.py varuna@${machines[i]}:/home/varuna/t-saathl/Varuna/varuna/
    # ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo mkdir -p /mnt/nitika/varuna_ckpts; sudo chown -R varuna /mnt/nitika"
    # scp -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Varuna/vmss_scripts/listen_preemption.py varuna@${machines[i]}:t-nisar/Varuna/vmss_scripts/
    # scp -i /home/varuna/.ssh/vdummy.pem ../apex_files/scaler.py varuna@${machines[i]}:apex/apex/amp/ 
    # scp -o "StrictHostKeyChecking no"  -i /home/varuna/.ssh/vdummy.pem /home/varuna/t-nisar/Megatron-base/Megatron-LM/megatron/training.py varuna@${machines[i]}:t-nisar/Megatron-base/Megatron-LM/megatron/
    i=$(($i+1))
done