machines=($(cat $1))
user="rahul"

nservers=${#machines[@]}

cd /home/varuna/t-saathl/Varuna/Megatron-LM

i=0 
pids=()
while [ $i -lt $nservers ]
do
    scp -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem startup.sh  $user@${machines[i]}: &
    pids+=($!) 
    i=$(($i+1))
done

for pid in ${pids[*]}; do
    wait $pid
done

i=0 
pids=()
while [ $i -lt $nservers ]
do
    ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem $user@${machines[i]} "nohup bash startup.sh > startup_out 2>startup_err &" &
    pids+=($!) 
    i=$(($i+1))
done

for pid in ${pids[*]}; do
    wait $pid
done

is_done=0
while [ $is_done == 0 ]
do
    sleep 30s
    is_done=1
    for machine in ${machines[*]}; do
        is_running=$(ssh -o ConnectTimeout=10 $machine "ps aux | grep startup.sh| grep -v grep| wc -l")
        if [ $is_running == 1 ]
        then
            is_done=0
            break
        fi
    done
done

echo "startup done!"