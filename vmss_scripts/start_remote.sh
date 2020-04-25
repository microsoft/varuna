machines=( $(az vmss nic list --vmss-name $1 --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
nservers=$3
ngpus=$2
nstages=$4
gpus_per_stage=$5
: '
machines=( "${machines[@]:0:29}" "${machines[@]:30}" )
#
#nservers=$(($nservers-1))
#gpus_per_stage=$(($gpus_per_stage-1))
echo $nservers
#'

if [ $# != 5 ]
then
    echo "need 5 arguments!!"
    exit
fi

# for change signal
echo $5>args

i=0
while [ $i != $nservers ]
do
    map="$i  ${machines[i]}"
    echo $map
    mid="run_varuna.py --node_rank $i --nservers $nservers --ngpus_per_server $ngpus --nstages $nstages --gpus_per_stage $gpus_per_stage"
    sudo ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem "varuna@${machines[i]}" "echo sshed; cd /home/varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT; sudo python3 $mid $5" > testouts/my_ssh_out_$i  2>testouts/my_ssh_err_$i &
    i=$(($i+1))
done

