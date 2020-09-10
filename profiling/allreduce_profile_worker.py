import torch
import os, sys
import time

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
replicas = int(sys.argv[3])
ring_size = world_size // replicas

print("Ring size:",ring_size)
print("World size:",world_size)
print("Rank", rank)

# for 1.5b
cp_size = 30740800
# 8.5b 36 partitions 384251904
factors = [2,3,4,6,8,12]
out_filename = "/home/varuna/gpt2-blob/perf_analysis_1.5b/all_red_rep_{}_ring_{}_clstrd_{}.txt".format(replicas, ring_size, rank)

out_file = open(out_filename,"w")

torch.distributed.init_process_group(rank=rank, world_size=world_size, backend='nccl')
# all reduce groups
all_reduce_group = None
for i in range(replicas):
    ranks = range(i*ring_size, (i+1)*ring_size)
    group = torch.distributed.new_group(ranks=ranks)
    if rank in ranks:
        all_reduce_group = group

warmup_allreduce = torch.ones(cp_size,dtype=torch.float16).cuda()
torch.distributed.all_reduce(warmup_allreduce,group=all_reduce_group)

out_file.write("Ring, # CPs, Load, time\n")
for i in factors:
    allreduce_size = cp_size * i
    avg_time = 0.0
    for _ in range(5):
        start_time = time.time()
        allred_tensor = torch.ones(allreduce_size,dtype=torch.float16).cuda()
        torch.distributed.all_reduce(allred_tensor,group=all_reduce_group)
        torch.cuda.synchronize()
        allred_time = time.time() - start_time
        print("ring", allred_tensor[0].item(), ", size", allreduce_size,": time", allred_time )
        avg_time += allred_time
    avg_time = avg_time / 5
    out_file.write("{}, {}, {}, {}\n".format(ring_size, i, allreduce_size, avg_time))

print("done!")
out_file.close()