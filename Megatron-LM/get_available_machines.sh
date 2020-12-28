cluster=${2:-"megatron"}
only_one=${3:-0}
subscription=${4:-"f3ebbda2-3d0f-468d-8e23-31a0796dcac1"}
group=${5:-"Varuna"}
print_unavailable=${1:-0}

machines=($(az vmss nic list --vmss-name $cluster --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
if [ $only_one -eq 0 ]
then
    machines+=($(az vmss nic list --vmss-name megatron_1 --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
    machines+=($(az vmss nic list --vmss-name megatron_2 --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
    #machines+=($(az vmss nic list --vmss-name megatron_3 --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
fi

reachable_machines=( )
unreachable_machines=( )
i=0
reachable_count=0
while [ $i -lt ${#machines[@]} ]
do
    # reachable=$(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "echo reachable")
    ping -c 1 -w 5 ${machines[i]} > ping.out
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
