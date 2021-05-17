
cd ~

ls fuse_connection2.cfg
if [ $? != 0 ]
then
    echo "accountName varunadata2" > fuse_connection2.cfg
    echo "accountKey b7R0zXBkC2404Vp1eQCr7O4C2JuGH/ymeDmEAa13tI2M2weI3GGHIl3a9NnDnHUEqCGsekLwLWTOXnjXj0VMvw==" >> fuse_connection2.cfg
    echo -n "containerName gpt2-blob" >> fuse_connection2.cfg
    sudo chmod 600 fuse_connection2.cfg
fi

blobmounted=$(ls gpt2-blob | wc -l)

if [ $blobmounted -lt 10 ]
then
    ls packages-microsoft-prod.deb
    if [ $? != 0 ]
    then
        wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
    fi
    i=0
    while [ $i -lt 50 ]
    do
        sudo dpkg -i packages-microsoft-prod.deb
        if [ $? != 0 ]
        then 
            echo "unsuccesful dpkg, waiting..."
            sleep 10s
        else
            echo "dpkg success!"
            break
        fi
        i=$(($i+1))
    done
    # sudo umount -f gpt2-blob
    sudo apt-get update
    sudo apt-get -y install blobfuse
    sudo blobfuse /home/rahul/gpt2-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/rahul/fuse_connection2.cfg -o allow_other
fi

codecopied=$(ls /home/rahul/Varuna | wc -l)
if [ $codecopied == 0 ]
then
    cp -r /home/rahul/gpt2-blob/single_gpu_longrunning/Varuna /home/rahul/
fi

export PATH="/home/rahul/anaconda3/bin:$PATH"

cd ~
python -c "from apex import amp; print('success')"
if [ $? != 0 ]
then
    sudo chown rahul -R /home/rahul/
    echo "copying apex..."
    cp -r /home/rahul/gpt2-blob/nvlamb_apex/apex /home/rahul/
    cd apex
    if [ $? == 0 ]
    then
        echo "installing apex..."
        pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    fi
fi

python -c "import varuna; print('success')"
if [ $? != 0 ]
then
    sudo chown rahul -R /home/rahul/Varuna
    cd /home/rahul/Varuna
    cd varuna
    g++ generate_schedule.cc -o genschedule
    cd ../tools/simulator
    g++ simulate-varuna-main.cc generate_schedule.cc simulate-varuna.cc -o simulate-varuna
    cd ../..
    python setup.py develop --user
    pip install pybind11
fi

path_exists=$(ls /home/rahul/Varuna/vmss_scripts/listen_preemption.py | wc -l)
already_running=$(ps aux | grep listen_preemption | grep -v grep | wc -l)
if [[ $path_exists == 1 ]] && [[ $already_running == 0 ]]
then
    echo "triggering listen script"
    cd ~/Varuna
    python vmss_scripts/listen_preemption.py > listen.out 2>listen.err &
fi

sudo mkdir -p /mnt/nitika
sudo chown rahul -R /mnt/nitika/
mkdir -p /mnt/nitika/varuna_ckpts/
# sudo dpkg --configure -a
# sudo apt-get -y install procmail