machines=(p40-gpu-0002 p40-gpu-0003 p40-gpu-0004 p40-gpu-0005 p100-gpu-0002 p100-gpu-0003 p100-gpu-0004 p100-gpu-0006)

nservers=`cat nservers`
i=1
while [ $i != $nservers ]
do
    echo ${machines[i]}
    # ssh "${machines[i]}" "kill -9 \$(ps -o pid,cmd |grep bert | awk -F ' ' '{print \$1}')"
    # ssh "${machines[i]}" "ps aux | grep python"
    # ssh ${machines[i]} "kill -9 \$(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^\$/d')"
    i=$(($i+1))
done