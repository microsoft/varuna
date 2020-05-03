machines=( $(az vmss nic list --vmss-name vmssdum --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
logs_machines=( )
nservers=$1
i=0
while [ $i != $nservers ]
do
    map="$i  ${machines[i]}"
    lscheck=$(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "ls DeepLearningExamples/PyTorch/LanguageModeling/BERT/ | grep of4")
    if [ ! -z "$lscheck" ]
    then
        #echo "$map has $lscheck"
	logs_machines+=(${machines[i]})
    fi
    i=$(($i+1))
done

#echo ${logs_machines[@]}
time_logs=( )
for machine in ${logs_machines[@]}
do
    exec 3< <(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@$machine" "python3 -c \"import time; print(time.time())\"" &)
    exec 4< <(ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${logs_machines[0]}" "python3 -c \"import time; print(time.time())\"" &)
    sleep 1s
    timeif=$(cat <&3)
    time0f=$(cat <&4)
    time_log=($timeif $time0f)
    #echo ${time_log[@]}
    time_logs+=(${time_log[@]})
    time_logs+=(";")
done

echo ${time_logs[@]}
