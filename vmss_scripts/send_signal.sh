ip_file=$1
local_pid_path=${2:"/tmp/varuna/local_parent_pid"}
machines=($(cat $ip_file))

echo "triggering stop signal"
i=0
while [ $i -lt ${#machines[@]} ]
do
    ssh ${machines[i]} "kill -10 \$(cat $local_pid_path)"
    i=$(($i+1))
done
