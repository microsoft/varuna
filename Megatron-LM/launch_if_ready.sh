
started=$(ps aux | grep catch_all | grep -v grep| wc -l)

if [ $started -gt 0 ]
then
    exit
fi

cd /home/varuna/t-saathl/Varuna

python3 vmss_scripts/scale_cluster.py 10 >> scale.out 2>>scale.err

cd Megatron-LM
machines=$(bash get_available_machines.sh | wc -l)
cd ..

if [ $machines -gt 8 ]
then
    python3 vmss_scripts/catch_all.py > catch.out 2>catch.err &
else
    az vmss scale --subscription f3ebbda2-3d0f-468d-8e23-31a0796dcac1 -g Varuna --name megatron --new-capacity 0
fi