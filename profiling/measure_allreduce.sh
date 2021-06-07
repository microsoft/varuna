reachable_machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))

reachable_count=${#reachable_machines[@]}
echo $reachable_count > nservers

num_servers=${1:-$reachable_count}

replicas=4
world_size=$(($replicas*$num_servers))
echo "world size $world_size"

master_addr=${reachable_machines[0]}
echo "master_addr: $master_addr"

cd /home/varuna/t-saathl/Varuna/Megatron-LM/
mkdir -p ssh_logs

i=0
while [ $i -lt $num_servers ]
do
    map="$i ${reachable_machines[i]}"
    echo $map
    # date > ssh_logs/my_ssh_err_$i
    # date > ssh_logs/my_ssh_out_$i
    # echo "$num_servers $i $master_addr" >> ssh_logs/my_ssh_err_$i
    r=0
    while [ $r -lt $replicas ]
    do
        # rank=$((($r*$num_servers) + $i))
        rank=$(($i*$replicas + $r))
        echo $rank
        allred_cmd="export PATH=\"/home/rahul/anaconda3/bin:\$PATH\"; cd Varuna/profiling;"
        allred_cmd="$allred_cmd CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 MASTER_ADDR=$master_addr MASTER_PORT=29500"
        allred_cmd="$allred_cmd python comm_profile_worker.py $rank $world_size allred_out_$rank 2>allred_err_$rank &"
        ssh -o StrictHostKeyChecking=no ${reachable_machines[i]} $allred_cmd
        r=$(($r+1))
    done
    i=$(($i+1))
done
