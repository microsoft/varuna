machines=($(cat /home/varuna/t-saathl/Varuna/Megatron-LM/available_machines.out))
nservers=${#machines[@]}

python3 ~/t-saathl/Varuna/vmss_scripts/slow_server.py > slow.out 2>slow.err &

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    scp -i /home/varuna/.ssh/vdummy.pem ~/t-saathl/Varuna/vmss_scripts/test_slow.py varuna@${machines[i]}:t-saathl/Varuna/vmss_scripts/ 
    for j in 0 1 2 3
    do
        ssh -i /home/varuna/.ssh/vdummy.pem varuna@${machines[i]} "export PATH=\"/home/varuna/anaconda3/bin:$PATH\"; CUDA_VISIBLE_DEVICES=$j python3 t-saathl/Varuna/vmss_scripts/test_slow.py" &
    done
    i=$(($i+1))
done

sleep 3m
sudo kill -9 $(ps aux |grep slow | awk -F ' ' '{print $2}')
