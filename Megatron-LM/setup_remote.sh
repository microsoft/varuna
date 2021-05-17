# machines=($(az vmss nic list --vmss-name megatron --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --resource-group Varuna --query [].{ip:ipConfigurations[0].privateIpAddress} --output tsv) )
# machines=($(cat $parentdir/Varuna/Megatron-LM/available_machines.out))
# parentdir="/home/varuna/t-saathl/tieweights"
parentdir="/home/varuna/t-saathl"
machines=($(cat $parentdir/Varuna/Megatron-LM/available_machines.out))
user="rahul"
# machines=(10.0.3.5)
nservers=${#machines[@]}


pythonpath="/home/$user/anaconda3/bin/python"
# pythonpath='python3.7'

# src="/home/varuna/.ssh/rconfig"
# dst="/home/$user/.ssh/config"
# src="/home/varuna/.ssh/vdummy.pem"
# dst="/home/$user/.ssh/."
# src="$parentdir/Varuna/"
# dst="/home/$user/"
src="$parentdir/Varuna/Megatron-LM/examples/pretrain_gpt2.sh"
dst="/home/$user/Varuna/Megatron-LM/examples/."
# src="$parentdir/Varuna/varuna/varuna.py"
# dst="/home/$user/Varuna/varuna/"
# src="$parentdir/Varuna/Megatron-LM/megatron/*.py"
# dst="/home/$user/Varuna/Megatron-LM/megatron/"
# src="$parentdir/Varuna/Megatron-LM/megatron/data/gpt2_dataset.py"
# dst="/home/$user/Varuna/Megatron-LM/megatron/data/"
# src="$parentdir/Varuna/Megatron-LM/pretrain_gpt2.py"
# dst="/home/$user/Varuna/Megatron-LM/"
# src="$parentdir/Varuna/Megatron-LM/megatron/training.py"
# dst="/home/$user/Varuna/Megatron-LM/megatron/"
# src="/home/varuna/t-saathl/Varuna/apex.patch"
# dst="/home/$user/apex/."
# src="$parentdir/Varuna/varuna/fused_lamb.py"
# dst="/home/$user/apex/apex/apex/optimizers/."
# src="/home/$user/t-saathl/Varuna/Megatron-LM-1dp-Megatron-LM-1dp-4mBS-stage0of9_0"
# dst="$parentdir/Varuna/Megatron-LM/logs/."
# src="/home/$user/Varuna/Megatron-LM/varuna_out"
# dst="$parentdir/Varuna/Megatron-LM/ssh_logs/."
# src="$parentdir/Varuna/Megatron-LM/profiles"
# dst="/home/$user/Varuna/Megatron-LM"
# src="$parentdir/nvlamb_apex/apex"
# dst="/home/$user/."
# src="/home/varuna/fuse_connection2.cfg"
# dst="/home/$user/."

# cmd="sudo chown rahul -R Varuna"
# cmd="export PATH=\"/home/rahul/anaconda3/bin:\$PATH\"; python -c 'import varuna' "
# cmd="git clone https://github.com/NVIDIA/apex"
# cmd="export PATH=\"/home/$user/anaconda3/bin:\$PATH\"; cd /home/$user/apex/; $pythonpath -m  pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./"
# cmd="sudo blobfuse /home/$user/gpt2-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/$user/fuse_connection2.cfg -o allow_other"
# cmd="sudo rm -r /home/$user/t-saathl/Varuna DeepLearningExamples Megatron-LM"
# cmd="ls apex | wc -l"
cmd="cd Varuna; $pythonpath setup.py develop"
# cmd="$pythonpath -c 'import varuna'"
# cmd="cd Varuna/tools/simulator; g++ simulate-varuna-main.cc generate_schedule.cc simulate-varuna.cc -o simulate-varuna"
# cmd="cp t-saathl/Varuna/varuna/fused_lamb.py apex/apex/apex/optimizers/fused_lamb.py"
# cmd="sudo chown $user apex/apex/apex/optimizers/*"
# cmd="sudo reboot"
# cmd="du -sh t-saathl/Varuna"
# cmd="sudo apt -y install blobfuse fuse"
# cmd="ifconfig | grep inet"
# cmd="mkdir t-saathl"
# cmd="cd apex;  git apply apex.patch"
# cmd="$pythonpath -m  pip install pybind11"
# cmd="cp -r ~/gpt2-blob/single_gpu_longrunning/Varuna /home/$user/"
# cmd="cd t-saathl/Varuna/Megatron-LM; rm vlogs-*"
# cmd="$pythonpath -c 'import apex'"
# cmd="mv t-saathl/Varuna t-saathl/nvaruna"
# cmd="mv apex oriapex"
# cmd="sudo rm -r t-saathl DeepLearningExamples; mkdir t-saathl"
# cmd="cd t-saathl/Varuna/Megatron-LM; sudo rm _tmp_*"
# cmd="cd t-saathl/Varuna/Megatron-LM; ls | grep _tmp"
# cmd="wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb; sudo dpkg -i packages-microsoft-prod.deb; sudo apt update"
# cmd="sudo rm /var/lib/dpkg/lock-frontend /var/cache/apt/archives/lock /var/lib/dpkg/lock"
# cmd="mkdir -p /home/$user/gpt2-blob; sudo chmod 600 fuse_connection2.cfg"

i=0
while [ $i -lt $nservers ]
do
    # echo $i ${machines[i]}

    # ssh -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem $user@${machines[i]} $cmd &

    scp -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem -r $src  $user@${machines[i]}:$dst &
    # scp -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem $user@${machines[i]}:Varuna/Megatron-LM/varuna_out ssh_logs/ssh_out_$i &
    # scp -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem $user@${machines[i]}:Varuna/Megatron-LM/varuna_err ssh_logs/ssh_err_$i &

    # scp -o StrictHostKeyChecking=no -i /home/varuna/.ssh/vdummy.pem -r $user@${machines[i]}:$src $dst 
    i=$(($i+1))
done