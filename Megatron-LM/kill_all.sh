# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
machines=($(cat /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out))

nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    ssh -o "StrictHostKeyChecking no" varuna@${machines[i]} -i /home/varuna/.ssh/vdummy.pem "sudo kill -9 \$(ps aux |grep pretrain | awk -F ' ' '{print \$2}')"
    i=$(($i+1))
done
echo "killed all"