machines=( $(az vmss nic list --vmss-name vmssdum --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )


if [ $# != 1 ]
then
    echo "need four arguments!!"
    exit
fi

# trigger signal on existing workers
orig_nservers=$1 # get this from - `cat nservers` in manager

i=0
while [ $i != $orig_nservers ]
do
    ssh -i ~/.ssh/id_rsa_gandiva "${machines[i]}" "cd /home/varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT; echo $1 > ngpus; echo $2 > nservers; echo $3 > nstages; echo $4 > gpus_per_stage; kill -10 \$(cat parent_process)"
    i=$(($i+1))
done

args=`cat args`