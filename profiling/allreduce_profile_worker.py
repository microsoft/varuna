import torch
import os, sys
import time

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
replicas = int(sys.argv[3])
ring_size = world_size // replicas

print("Ring size:",ring_size)
print("World size:",world_size)
print("Rank", rank, flush = True)

# for 1.5b
cp_size = 44261760
embed_size = 50304*1920
# 8.5b 36 partitions 384251904
factors = [1,2,3,6,9,18]
out_filename = "/home/varuna/gpt2-blob/perf_analysis_2.5b/all_red_rep_{}_ring_{}_clstrd_{}.txt".format(replicas, ring_size, rank)
print(out_filename, flush=True)

out_file = open(out_filename,"w")

torch.distributed.init_process_group(rank=rank, world_size=world_size, backend='nccl')
# all reduce groups
all_reduce_group = None
for i in range(replicas):
    ranks = range(i*ring_size, (i+1)*ring_size)
    group = torch.distributed.new_group(ranks=ranks)
    if rank in ranks:
        all_reduce_group = group

embed_group = None
for i in range(replicas):
    ranks = [i*ring_size, ((i+1)*ring_size) - 1]
    print(ranks)
    group = torch.distributed.new_group(ranks=ranks)
    if rank in ranks:
        embed_group = group

print(embed_group is None)

warmup_allreduce = torch.ones(cp_size,dtype=torch.float16).cuda()
torch.distributed.all_reduce(warmup_allreduce,group=all_reduce_group)
del warmup_allreduce

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
        print("ring", allred_tensor[0].item(), ", size", allreduce_size,": time", allred_time,flush=True)
        avg_time += allred_time
    print("torch mem", torch.cuda.memory_allocated(), torch.cuda.memory_cached())
    avg_time = avg_time / 5
    out_file.write("{}, {}, {}, {}\n".format(ring_size, i, allreduce_size, avg_time))
    del allred_tensor
    torch.cuda.empty_cache()

out_file.close()

out_file = open("/home/varuna/gpt2-blob/perf_analysis_2.5b/embed_all_red_{}".format(rank),"a")

if embed_group is not None:
    avg_time = 0.0
    for _ in range(5):
        start_time = time.time()
        allred_tensor = torch.ones(embed_size).cuda()
        torch.distributed.all_reduce(allred_tensor,group=embed_group)
        torch.cuda.synchronize()
        allred_time = time.time() - start_time
        avg_time += allred_time
    avg_time = avg_time / 5
    out_file.write("world: {}; embed allred {} {}\n".format(world_size, embed_size, avg_time))

# torch.distributed.all_reduce(torch.cuda.IntTensor([0]))
print("done!")
out_file.close()