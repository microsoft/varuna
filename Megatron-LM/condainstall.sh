if [ $# != 1 ]
then
    echo "need 5 arguments!!"
    exit
fi

#machines=( $(az vmss nic list --vmss-name $2 --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))
#machines=($(cat /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out))

nservers=$1
i=0
while [ $i != $nservers ]
do
    map="$i  ${machines[i]}"
    echo $map
    ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "sudo umount /home/varuna/bert-large-blob" 
    ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "sudo blobfuse /home/varuna/bert-large-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/varuna/fuse_connection.cfg -o allow_other" 
    ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "sudo cp /home/varuna/bert-large-blob/install.sh /home/varuna/.; sudo chmod +x install.sh; ./install.sh" &
    i=$(($i+1))
done

