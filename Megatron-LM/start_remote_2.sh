reachable_machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))

reachable_count=${#reachable_machines[@]}
echo $reachable_count > nservers

if [ $# == 0 ]
then
    echo "Need cmd with args!"
    exit
fi

replicas=4 
master_addr=${reachable_machines[0]}
echo "master_addr: $master_addr"

world_size=$(($reachable_count*$replicas))
global_envs="GLOO_SOCKET_IFNAME=eth0 MASTER_PORT=29500 WORLD_SIZE=$world_size MASTER_ADDR=$master_addr"
conda_cmd="export PATH=\"/home/varuna/anaconda3/bin:\$PATH\""
cmd_with_args="$@"
codepath=`pwd`
echo $cmd_with_args
mkdir -p ssh_logs

i=0
while [ $i -lt ${#reachable_machines[@]} ]
do
    map="$i ${reachable_machines[i]}"
    echo $map
    r=0
    while [ $r -lt $replicas ]
    do
        # rank=$((($r*$num_servers) + $i))
        rank=$(($i*$replicas + $r))
        echo $rank
        cmd="$conda_cmd; cd $codepath; $global_envs RANK=$rank LOCAL_RANK=$r CUDA_VISIBLE_DEVICES=$r $cmd_with_args" 
        ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${reachable_machines[i]}" \
                "$cmd" > ssh_logs/my_ssh_out_$rank 2>ssh_logs/my_ssh_err_$rank &
        r=$(($r+1))
    done
    i=$(($i+1))
done
