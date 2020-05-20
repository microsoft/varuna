reachable_machines=($(cat /home/varuna/t-saathl/mega1_5b/Megatron-LM/available_machines.out))

reachable_count=${#reachable_machines[@]}
echo $reachable_count > nservers

if [ $# != 1 ]
then
    echo "Need ckpt arg!"
    exit
fi

ckpt=$1

master_addr=${reachable_machines[0]}
echo "master_addr: $master_addr"

cd /home/varuna/t-saathl/mega1_5b/Megatron-LM/
mkdir -p testouts
i=0
while [ $i -lt ${#reachable_machines[@]} ]
do
    map="$i ${reachable_machines[i]}"
    echo $map
    sudo ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${reachable_machines[i]}" "echo sshed; cd /home/varuna/t-saathl/Varuna/Megatron-LM ; bash examples/pretrain_gpt2.sh $reachable_count $i $master_addr $ckpt" >> testouts/my_ssh_out_$i  2>testouts/my_ssh_err_$i &
    i=$(($i+1))
done