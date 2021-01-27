ip_list=${1}

if [ $# == 0 ]
then
    echo "Need IP list!"
    exit
fi
if [ ! -f $ip_list ]
then
    echo "IP list path invalid"
    exit
fi

machines=($(cat $ip_list))

reachable_machines=( )
unreachable_machines=( )
i=0
reachable_count=0
while [ $i -lt ${#machines[@]} ]
do
    ping -c 1 -w 5 ${machines[i]} > ping.out
    reachable=$?
    if [ $reachable == 0 ]
    then
        reachable_machines+=(${machines[i]})
        reachable_count=$(($reachable_count+1))
    else
        unreachable_machines+=(${machines[i]})
    fi
    i=$(($i+1))
done

rm ping.out
i=0
while [ $i -lt $reachable_count ]
do
    echo ${reachable_machines[i]}
    i=$(($i+1))
done

if [ $print_unavailable == 1 ]
then
    echo "unreachable"
    i=0
    while [ $i -lt ${#unreachable_machines[@]} ]
    do
        echo ${unreachable_machines[i]}
        i=$(($i+1))
    done
fi
