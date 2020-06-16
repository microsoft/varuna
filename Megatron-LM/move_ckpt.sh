
source=$1
destination=$2
update_tracker=$3
parent_pid=$4

# if [ $# -lt 3 ]
# fi

error=0

mv $source $destination
error=$(( $error -o $? ))

if [ $error == 0 ]
then
    counter=$(cat /home/varuna/local_ckpt_tracker.txt)
    counter=$(($counter+1))
    echo $counter > /home/varuna/local_ckpt_tracker.txt
fi

error=$(( $error -o $? ))

if [ $error == 1 ]
then
    kill -12 $parent_pid
fi
