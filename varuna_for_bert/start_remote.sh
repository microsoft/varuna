machines=(p100-gpu-0001 p100-gpu-0002 p100-gpu-0003 p100-gpu-0004 p100-gpu-0006 p100-gpu-0007)

if [ $# != 3 ]
then
    echo "need 3 arguments!!"
    exit
fi

masteraddr="10.4.0.29"
nservers=$1
executable=$2

cd ~/t-nisar/Varuna/varuna_for_bert; source activate varuna;

i=1
while [ $i != $1 ]
do
    echo $i
    mid="launch_bert.py --node_rank $i --nservers $1"
    ssh "${machines[i]}" "echo sshed; cd ~/t-nisar/Varuna/varuna_for_bert; source activate varuna; echo \$SQUAD_DIR; $executable $mid $3" > ssh_out_$i &
    i=$(($i+1))
done
mid="launch_bert.py --node_rank 0 --nservers $1"
$executable $mid $3
# ssh p100-gpu-0003 "cd ~/t-nisar/Varuna/varuna_for_bert; source activate varuna;