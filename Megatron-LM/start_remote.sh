# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
reachable_machines=($(cat /home/varuna/t-nisar/Megatron-LM/available_machines.out))


reachable_count=${#reachable_machines[@]}
echo $reachable_count > nservers

if [ $# != 1 ]
then
    echo "Need ckpt arg!"
    exit
fi

ckpt=$1

# reachable_machines=( )
# i=0
# reachable_count=0
# while [ $reachable_count -lt $nservers -a $i -lt ${#machines[@]} ]
# do
#     reachable=$(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "echo reachable")
#     if [ "$reachable" == "reachable" ]
#     then
#         reachable_machines+=(${machines[i]})
#         reachable_count=$(($reachable_count+1))
#     else
#         echo "$i ${machines[i]} not reachable"
#     fi
#     i=$(($i+1))
# done

# if [ $reachable_count == 0 ]
# then
#     echo "No reachable machines found!"
#     exit
# fi

# echo "found $reachable_count machines to run!"

master_addr=${reachable_machines[0]}
echo "master_addr: $master_addr"

i=0
while [ $i -lt ${#reachable_machines[@]} ]
do
    map="$i ${reachable_machines[i]}"
    echo $map
    sudo ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${reachable_machines[i]}" "echo sshed; cd /home/varuna/t-nisar/Megatron-LM ; bash examples/pretrain_gpt2.sh $reachable_count $i $master_addr $ckpt" >> my_ssh_out_${reachable_machines[i]}  2>>my_ssh_err_${reachable_machines[i]} &
    i=$(($i+1))
done