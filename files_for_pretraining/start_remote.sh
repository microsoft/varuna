machines=(v100-gpu-1 v100-gpu-2 v100-gpu-3 v100-gpu-4 v100-gpu-5)

if [ $# != 5 ]
then
    echo "need 5 arguments!!"
    exit
fi

nservers=$2
ngpus=$1
nstages=$3
gpus_per_stage=$4

# for change signal
echo $5>args

i=1
while [ $i != $nservers ]
do
    echo $i
    mid="run_varuna.py --node_rank $i --nservers $nservers --ngpus_per_server $ngpus --nstages $nstages --gpus_per_stage $gpus_per_stage"    
    ssh -i ~/.ssh/id_rsa_gandiva "${machines[i]}" "echo sshed; docker exec -w /data/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT --privileged nika python $mid $5" > my_ssh_out_$i  2>my_ssh_err_$i &
    i=$(($i+1))
done
mid="run_varuna.py --node_rank 0 --nservers $nservers --ngpus_per_server $ngpus --nstages $nstages --gpus_per_stage $gpus_per_stage"
docker exec -w /data/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT --privileged nika python $mid $5