machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv | shuf) )

reachable_machines=( )
i=0
reachable_count=0
while [ $i -lt ${#machines[@]} ]
do
    reachable=$(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "echo reachable")
    if [ "$reachable" == "reachable" ]
    then
        reachable_machines+=(${machines[i]})
        reachable_count=$(($reachable_count+1))
    fi
    i=$(($i+1))
done

# echo "$reachable_count reachable"
# morphable=$(($reachable_count-2))
available=$((20 + RANDOM))
available=$(($available % $reachable_count))
# echo "$available available"
i=0
while [ $i -lt $reachable_count ]
do
    echo ${reachable_machines[i]}
    i=$(($i+1))
done

