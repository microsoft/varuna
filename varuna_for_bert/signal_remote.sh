machines=(p40-gpu-0002 p40-gpu-0004 p40-gpu-0005 p40-gpu-0007 p100-gpu-0001 p100-gpu-0002 p100-gpu-0003 p100-gpu-0004)

if [ $# != 2 ]
then
    echo "need two arguments!!"
    exit
fi

orig_nservers=`cat nservers`

i=1
while [ $i != $orig_nservers ]
do
    ssh "${machines[i]}" "cd ~/t-nisar/Varuna/varuna_for_bert; echo $1 > ngpus; echo $2 > nservers; echo try2kill; pid=\`cat parent_process\`; kill -10 \$pid"
    i=$(($i+1))
done

args=`cat args`

while [ $i -lt $2 ]
do
    mid="launch_bert.py --node_rank $i --nservers $1 --ngpus_per_server $2"
    ssh "${machines[i]}" "echo sshed; cd ~/t-nisar/Varuna/varuna_for_bert; source ~/anaconda3/bin/activate varuna; echo \$SQUAD_DIR; python $mid $args" > ssh_out_$i  2>ssh_err_$i &
    i=$(($i+1))
done

echo $1 > ngpus
echo $2 > nservers
pid=`cat parent_process`
kill -10 $pid
 