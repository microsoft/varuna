reachable_machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))
reachable_machines=( "${reachable_machines[@]:0:10}" )

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

cd /home/varuna/t-saathl/Varuna/Megatron-LM/
mkdir -p saathl_logs
mkdir -p saathl_acc_logs

i=0
while [ $i -lt ${#reachable_machines[@]} ]
do
    map="$i ${reachable_machines[i]}"
    echo $map

    # acc logs
    cat saathl_logs/my_ssh_err_$i >> saathl_acc_logs/my_ssh_err_$i
    cat saathl_logs/my_ssh_out_$i >> saathl_acc_logs/my_ssh_out_$i

    date > saathl_logs/my_ssh_err_$i
    echo "$reachable_count $i $master_addr $ckpt" >> saathl_logs/my_ssh_err_$i
    sudo ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${reachable_machines[i]}" "echo sshed; cd /home/varuna/t-saathl/Varuna/Megatron-LM ; bash examples/pretrain_gpt2.sh $reachable_count $i $master_addr $ckpt" > saathl_logs/my_ssh_out_$i  2>>saathl_logs/my_ssh_err_$i &
    i=$(($i+1))
done
