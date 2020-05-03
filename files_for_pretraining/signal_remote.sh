machines=(v100-gpu-1 v100-gpu-2 v100-gpu-3 v100-gpu-4 v100-gpu-5 v100-gpu-6)

cd /mnt/sdb1/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT

if [ $# != 4 ]
then
    echo "need four arguments!!"
    exit
fi

# trigger signal on existing workers
orig_nservers=`cat nservers`

i=1
while [ $i != $orig_nservers ]
do
    ssh -i ~/.ssh/id_rsa_gandiva "${machines[i]}" "docker exec -w /data/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT nika sh -c \"echo $1 > ngpus; echo $2 > nservers; echo $3 > nstages; echo $4 > gpus_per_stage; kill -10 \\\$(cat parent_process)\""
    i=$(($i+1))
done

echo $1 > ngpus; echo $2 > nservers; echo $3 > nstages; echo $4 > gpus_per_stage; 
pid=`cat parent_process`
docker exec -w /data/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT nika kill -10 $pid


args=`cat args`

# wait for checkpointing and exit 
prev_done="notdone"
while [ $prev_done != "done" ]
do
    prev_done=`cat prev_job_done`
done

echo "AND FINALLY DONE"

resume_step=`cat resume_step`

# spawn new
while [ $i -lt $2 ]
do
    mid="run_varuna.py --node_rank $i --ngpus_per_server $1 --nservers $2 --nstages $3 --gpus_per_stage $4"
    ssh -i ~/.ssh/id_rsa_gandiva "${machines[i]}" "echo sshed; docker exec -w /data/t-nisar/DeepLearningExamples/PyTorch/LanguageModeling/BERT  nika python $mid $args --resume_from_checkpoint --resume_step $resume_step" >> my_ssh_out_$i  2>>my_ssh_err_$i &
    i=$(($i+1))
done

