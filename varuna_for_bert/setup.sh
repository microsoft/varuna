cd ~
mkdir t-nisar
cd t-nisar
git clone https://adaptivednn.visualstudio.com/Varuna/_git/Varuna

conda create -n varuna python=3.6.2
source activate varuna
pip install torch==1.3.1
pip install tqdm tensorboardX pathlib

git clone https://github.com/huggingface/transformers/
cd Varuna
git checkout nitika/profiling
cd ..
cp Varuna/varuna_for_bert/modeling_bert.py transformers/src/transformers/modeling_bert.py
mkdir -p ~/t-nisar/data/squad
scp -r core@p40-gpu-0002:t-nisar/data/squad ~/t-nisar/data/

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
echo "export SQUAD_DIR=/home/core/t-nisar/data/squad" > ./etc/conda/activate.d/env_vars.sh
echo "unset SQUAD_DIR" > ./etc/conda/deactivate.d/env_vars.sh

cd ~
wget https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse
sudo mkdir /mnt/resource/blobfusetmp -p
sudo chown core /mnt/resource/blobfusetmp
touch ~/fuse_connection.cfg
chmod 600 fuse_connection.cfg
mkdir ~/myblobcontainer
sudo blobfuse ~/myblobcontainer --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/core/fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other -o uid=1000 -o gid=1000

cd ~/t-nisar/transformers
python setup.py develop

cd ~/t-nisar/Varuna
python setup.py develop
cd varuna
g++ generate_schedule.cc -o genschedule
cd ..
cd varuna_for_bert
mkdir training_output
mkdir train_reports

