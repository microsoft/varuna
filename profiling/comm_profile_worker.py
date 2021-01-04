
import torch
import sys, time
import math
import torch.distributed as dist
from torch.multiprocessing import Process
from queue import Queue
from threading import Thread

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
local_rank = int(sys.argv[3])

print("Rank",rank)
print("World size", world_size,flush=True)

prev_size = torch.Size([1,1024,3072])
next_size = torch.Size([1,1024,3072])

batch_size = 512
microBSs = range(1,20)

torch.distributed.init_process_group(rank=rank, world_size=world_size, backend='gloo')

next_rank = None; prev_rank = None
if rank < world_size - 1:
    next_rank = rank + 1
if rank > 0:
    prev_rank = rank - 1

def receiver(recv_rank, mBS, recv_times):
    print("Start reciever from", recv_rank)
    chunks = math.ceil(batch_size/ mBS)
    dtype = torch.float16
    recv_shape = list(prev_size) if recv_rank == prev_rank else list(next_size)
    recv_shape[0] = mBS
    recv_shape = torch.Size(recv_shape)
    recv_handles = Queue()

    for _ in range(chunks):
        acts_tensor = torch.ones(recv_shape, dtype=dtype)
        start_time = time.time()
        # print("recv from", recv_rank, flush=True)
        handle = dist.irecv(acts_tensor, src=recv_rank)
        recv_handles.put((handle, start_time))
        if recv_handles.qsize() > 4:
            handle, start_time = recv_handles.get()
            handle.wait()
            recv_times.append(time.time() - start_time)

    while not recv_handles.empty():
        handle, start_time = recv_handles.get()
        handle.wait()
        recv_times.append(time.time() - start_time)
    

def sender(send_rank, mBS, send_times):
    print("Start sender to",send_rank)
    chunks = math.ceil(batch_size/ mBS)
    dtype = torch.float16
    send_shape = list(prev_size) if send_rank == prev_rank else list(next_size)
    send_shape[0] = mBS
    send_shape = torch.Size(send_shape)
    send_handles = Queue()

    for _ in range(chunks):
        output_acts = torch.ones(send_shape, dtype=dtype)
        start_time = time.time()
        # print("send to", send_rank, flush=True)
        handle = dist.isend(output_acts, dst=send_rank)
        send_handles.put((handle, start_time))
        if send_handles.qsize() > 4:
            handle, start_time = send_handles.get()
            handle.wait()
            send_times.append(time.time() - start_time)
        # count -= 1
    
    while not send_handles.empty():
        handle, start_time = send_handles.get()
        handle.wait()
        send_times.append(time.time() - start_time)


out_file = "/home/varuna/gpt2-blob/perf_analysis_8.3b/send_times_clstrd_new_{}".format(rank)
out_file = open(out_file,"w")
out_file.write("mBS, acts send time, acts recv time, grads send time, grads recv time\n")

for mBS in microBSs:
    print("mBS",mBS, flush = True)

    acts_recv_times = []; grads_recv_times = []
    acts_send_times = []; grads_send_times = []
    acts_receive_thread = grads_receive_thread = None
    acts_send_thread = grads_send_thread = None

    if prev_rank is not None:
        acts_receive_thread = Thread(target=receiver, args=(prev_rank, mBS, acts_recv_times))
        acts_receive_thread.daemon=True
        acts_receive_thread.start()

        grads_send_thread = Thread(target=sender, args=(prev_rank, mBS, grads_send_times))
        grads_send_thread.daemon=True
        grads_send_thread.start()

    if next_rank is not None:
        grads_receive_thread = Thread(target=receiver, args=(next_rank, mBS, grads_recv_times))
        grads_receive_thread.daemon=True
        grads_receive_thread.start()

        acts_send_thread = Thread(target=sender, args=(next_rank, mBS, acts_send_times))
        acts_send_thread.daemon=True
        acts_send_thread.start()

    print("threads started",flush=True)
    if acts_receive_thread is not None:
        acts_receive_thread.join()
    if grads_receive_thread is not None:
        grads_receive_thread.join()
    if acts_send_thread is not None:
        acts_send_thread.join()
    if grads_send_thread is not None:
        grads_send_thread.join()

    acts_recv_times = acts_recv_times[5:]
    acts_send_times = acts_send_times[5:]
    grads_recv_times = grads_recv_times[5:]
    grads_send_times = grads_send_times[5:]

    print(rank,len(acts_recv_times), len(acts_send_times))

    # print(acts_send_times)
    out_file.write("micro BS {}:\n".format(mBS))

    act_send_time = act_recv_time = 0
    grad_send_time = grad_recv_time = 0
    if len(acts_send_times) > 0:
        act_send_time =  sum(acts_send_times)/len(acts_send_times)
        print(rank, "avg send time",act_send_time)
        print(rank, "min-max send time", min(acts_send_times), max(acts_send_times) )
        out_file.write("actsend: {}\n".format(",".join([str(t) for t in acts_send_times])))
    if len(acts_recv_times) > 0:
        act_recv_time =  sum(acts_recv_times)/len(acts_recv_times)
        print(rank, "avg recv time", act_recv_time)
        print(rank, "min-max recv time", min(acts_recv_times), max(acts_recv_times) )
        # out_file.write("actrecv: {}\n".format(",".join([str(t) for t in acts_recv_times]))
    if len(grads_send_times) > 0:
        grad_send_time =  sum(grads_send_times)/len(grads_send_times)
        print(rank, "avg send time",grad_send_time)
        print(rank, "min-max send time", min(grads_send_times), max(grads_send_times) )
        out_file.write("gradsend: {}\n".format(",".join([str(t) for t in grads_send_times])))
    if len(grads_recv_times) > 0:
        grad_recv_time =  sum(grads_recv_times)/len(grads_recv_times)
        print(rank, "avg recv time", grad_recv_time)
        print(rank, "min-max recv time", min(grads_recv_times), max(grads_recv_times) )
        # out_file.write("gradrecv: {}\n".format(",".join([str(t) for t in grads_recv_times]))

    # out_file.write("{}, {}, {}, {}, {}\n".format(mBS, act_send_time, act_recv_time, grad_send_time, grad_recv_time))
    

out_file.close()
