# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
machines=($(cat /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out))

echo "triggering stop signal"
i=0
while [ $i -lt ${#machines[@]} ]
do
    ssh -o "StrictHostKeyChecking no" -i ~/.ssh/vdummy.pem "varuna@${machines[i]}" "cd t-nisar/Varuna/Megatron-LM; echo \"0\" >nservers; kill -10 \$(cat parent_process)"
    i=$(($i+1))
done
