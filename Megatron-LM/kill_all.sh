machines=($(cat /home/varuna/t-saathl/tieweights/Varuna/Megatron-LM/available_machines.out))

nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo kill -9 \$(ps aux |grep pretrain | awk -F ' ' '{print \$2}')" 
    i=$(($i+1))
done
