print_unavailable=${1:-0}
cluster=${2:-"single_gpu_spots_1"}
only_one=${3:-0}
ignore_scaling=${4:-0}
subscription=${5:-"f3ebbda2-3d0f-468d-8e23-31a0796dcac1"}
group=${6:-"Varuna"}
user="rahul"

machines=($(az vmss nic list --vmss-name $cluster --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
# machines+=($(az vmss nic list --vmss-name single_gpu_spots --subscription $subscription --resource-group $group --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
scaling=$(ps aux | grep scale_cluster| grep -v grep| wc -l)
if [[ $scaling != 0 ]] && [[ $ignore_scaling == 0 ]]
then
    machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))
fi

slow_machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/slow_machines.out))
reachable_machines=( )
unreachable_machines=( )

reachable_count=0

is_reachable=()
is_slow=()
i=0
while [ $i -lt ${#machines[@]} ]
do
    is_reachable+=(1)
    is_slow+=(0)
    i=$(($i+1))
done

cd /home/varuna/t-saathl/Varuna/Megatron-LM
mkdir -p tmp
i=0
pids=()
ts=$(($(date +%s%N)/1000000))
# echo $ts 1>&2
while [ $i -lt ${#machines[@]} ]
do
    # echo ${machines[i]} 1>&2
    # echo "tmp/ping_{$ts}_$i" 1>&2
    ping_cmd="ping -c 1 -w 5 ${machines[i]}"
    # ssh_cmd="timeout 10 ssh -o StrictHostKeyChecking=no -i ~/.ssh/vdummy.pem $user@${machines[i]} echo -n ''"
    # $ssh_cmd 1>&2
    { $ping_cmd >tmp/ping_{$ts}_$i; echo $? > tmp/ping_{$ts}_$i; } &
    pids+=($!) 
    slow=0
    for slow_ip in ${slow_machines[*]}; do
        if [ $slow_ip == ${machines[i]} ]
        then
            slow=1
            break
        fi
    done
    is_slow[i]=$slow
    i=$(($i+1))
done

# echo "waiting" 1>&2
i=0
for pid in ${pids[*]}; do
    # echo "waiting for ${machines[i]}" 1>&2
    wait $pid
    i=$(($i+1))
done
# echo "wait done" 1>&2
i=0
while [ $i -lt ${#machines[@]} ]
do
    is_reachable[i]=$(<tmp/ping_{$ts}_$i)
    # echo ${machines[i]} ${is_reachable[i]}
    rm tmp/ping_{$ts}_$i
    if [[ ${is_reachable[i]} == 0 ]] && [[ ${is_slow[i]} == 0 ]]
    then
        echo ${machines[i]}
    else
        unreachable_machines+=(${machines[i]})
    fi
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
