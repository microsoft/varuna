cluster=${1:-"megatron"}
print_unavailable=${2:-0}

machines=($(az vmss nic list --vmss-name $cluster --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )

reachable_machines=( )
unreachable_machines=( )
i=0
reachable_count=0
while [ $i -lt ${#machines[@]} ]
do
    # reachable=$(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "echo reachable")
    ping -c 1 ${machines[i]} > ping.out
    reachable=$?
    if [ $reachable == 0 ]
    then
        reachable_machines+=(${machines[i]})
        reachable_count=$(($reachable_count+1))
    else
        unreachable_machines+=(${machines[i]})
    fi
    i=$(($i+1))
done

rm ping.out
i=0
while [ $i -lt $reachable_count ]
do
    echo ${reachable_machines[i]}
    i=$(($i+1))
done

if [ $print_unavailable == 1 ]
then
    echo "unreachable"
    i=0
    while [ $i -lt ${#unreachable_machines[@]} ]
    do
        echo ${unreachable_machines[i]}
        i=$(($i+1))
    done
fi
