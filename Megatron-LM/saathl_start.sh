# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
reachable_machines=($(cat /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out))
reachable_machines=( "${reachable_machines[@]:0:74}" )

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

cd ~/t-nisar/Varuna/Megatron-LM/

i=0
while [ $i -lt ${#reachable_machines[@]} ]
do
    map="$i ${reachable_machines[i]}"
    echo $map
    sudo ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${reachable_machines[i]}" "echo sshed; cd /home/varuna/t-nisar/Varuna/Megatron-LM ; bash examples/pretrain_gpt2.sh $reachable_count $i $master_addr $ckpt" >> ssh_logs/my_ssh_out_$i  2>ssh_logs/my_ssh_err_$i &
    i=$(($i+1))
done
