machines=(p100-gpu-0001 p100-gpu-0002 p100-gpu-0003 p100-gpu-0004 p100-gpu-0006 p100-gpu-0007)

masteraddr="10.4.0.29"

if [ $# != 3 ]
then
    echo "need 3 arguments!!"
    exit
fi

i=1
while [ $i != $1 ]
do
    ssh "${machines[i]}" "cd ~/t-nisar/Varuna/varuna_for_bert; echo $2 > ngpus; echo $3 > nservers; echo try2kill; pid=\`cat parent_process\`; kill -10 \$pid"
    i=$(($i+1))
done

echo $2 > ngpus
echo $3 > nservers
pid=`cat parent_process`
kill -10 $pid

# bash signal_remote.sh 2 2 2 
 