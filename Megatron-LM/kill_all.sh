ip_file=${1:-"/home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out"}
machines=($(cat $ip_file))
nservers=${#machines[@]}
user="rahul"

i=0
pids=()
while [ $i -lt $nservers ]
do
    # echo $i ${machines[i]}
    timeout 10 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $user@${machines[i]} "sudo pkill -f pretrain" &
    pids+=($!) 
    i=$(($i+1))
done

for pid in ${pids[*]}; do
    wait $pid
done

echo "job killed!"
