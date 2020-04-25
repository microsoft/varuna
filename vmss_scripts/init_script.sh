#!/bin/sh
# setup blob fuse for data set and checkpoints
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=16g tmpfs /mnt/ramdisk
sudo mkdir /mnt/ramdisk/blobfusetmp
sudo chown varuna /mnt/ramdisk/blobfusetmp
mkdir -p /home/varuna/bert-large-blob
sudo blobfuse /home/varuna/bert-large-blob --tmp-path=/mnt/ramkdisk/blobfusetmp --config-file=/home/varuna/fuse_connection.cfg -o allow_other

# download code repository
cd /home/varuna
rm -r DeepLearningExamples
cp -r /home/varuna/bert-large-blob/DeepLearningExamples /home/varuna/.

# setup Varuna
cd /home/varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT/Varuna; python3 setup.py develop

# start script to listen to pre-emptions
# python3 listen_preemption.py > preemptout &
