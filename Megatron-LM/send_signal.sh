machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))

echo "triggering stop signal"
i=0
cmd="cd Varuna/Megatron-LM; echo \"0\" >nservers; pkill -10 -f run_varuna"
while [ $i -lt ${#machines[@]} ]
do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/vdummy.pem rahul@${machines[i]} $cmd &
    i=$(($i+1))
done
echo "stopped jobs!"