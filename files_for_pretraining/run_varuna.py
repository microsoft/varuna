import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
import math
# import atexit

local_rank_to_device = [0,1,2,3]

# different for diff kinds of GPUs
MAX_GPU_MEM = 16280000000

def calculate_config(args):
    # world size in terms of number of processes
    gpus_available = args.ngpus_per_server * args.nservers
    gpus_per_stage = (gpus_available // args.nstages) if args.gpus_per_stage == 0 else args.gpus_per_stage
    dist_world_size = gpus_per_stage * args.nstages
    assert(dist_world_size < gpus_available, "Too many gpus_per_stage!")
    unused_gpus = (gpus_available - dist_world_size)

    # one whole server is unused
    if unused_gpus > args.ngpus_per_server:
        raise ValueError("Wrong number of servers - too many unused GPUs")

    stage_to_rank_map = {}
    rank_to_stage_map = {}

    # if args.placement=='clustered':
    #     for i in range(args.nstages):
    #         stage_to_rank_map[i]= range(i,dist_world_size,args.nstages)
    # else:
    for i in range(0,dist_world_size,gpus_per_stage):
        stage_to_rank_map[int(i//gpus_per_stage)] = range(i,i+gpus_per_stage)
    stage_to_rank_map_str = ""
    for stage in stage_to_rank_map:
        ranks = ",".join([str(r) for r in stage_to_rank_map[stage]])
        stage_to_rank_map_str += (ranks + ";")

    # batch size should be divisible by num of data parallel workers
    per_gpu_batch_size = args.batch_size // gpus_per_stage
    train_batch_size = per_gpu_batch_size * gpus_per_stage
    
    # per_gpu_micro_batch_size = math.ceil(per_gpu_batch_size / (1.0 * args.chunks))
    # gradient_accumulation_steps = 1
    # max_gradient_accumulation_steps = per_gpu_micro_batch_size // max_micro_batch_size_per_gpu

    # # shouldn't ceil(max_grad_acc_steps) just be the actual one we use??
    # while per_gpu_micro_batch_size > max_micro_batch_size_per_gpu and gradient_accumulation_steps <= max_gradient_accumulation_steps :
    #     gradient_accumulation_steps += 1
    #     per_gpu_micro_batch_size_ = per_gpu_micro_batch_size // gradient_accumulation_steps 
    #     if per_gpu_micro_batch_size_ <= max_micro_batch_size_per_gpu:
    #         per_gpu_micro_batch_size = per_gpu_micro_batch_size_
    #         break

    # # for pipelining, we don't really need gradient accumalation steps - just increase the number of chunks
    # # we want to keep the micro BS same
    # args.chunks = args.chunks * gradient_accumulation_steps

    first_rank_in_server = args.node_rank * args.ngpus_per_server
    if args.node_rank == args.nservers - 1:
        ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server - unused_gpus)
    else:
        ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)

    print("Config:")
    print("train batch size:",args.batch_size)
    print("partitions:", args.nstages)
    print("chunk_size:", args.chunk_size)
    print("data depth:", gpus_per_stage)
    print("stage to rank map:", stage_to_rank_map_str)

    return dist_world_size, stage_to_rank_map_str, ranks_in_server, train_batch_size
    


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="BERT PyTorch model parallel training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--ngpus_per_server", type=int, default=4,
                        help="The desired number of GPUs per server. Each process can be bound to a single GPU.")

    parser.add_argument("--nservers", type=int, default=1,
                        help="The total number of nodes.")

    parser.add_argument("--node_rank", type=int, default=0,
                        help="Rank of node amongst servers.")

    parser.add_argument("--nstages", type=int, required = True,
                        help="Depth of pipeline (number of stages)")

    parser.add_argument("--batch_size", required=True, type=int,
                        help="Total effective batch size required")

    parser.add_argument("--profile", required=True, type=str,
                        help="CSV file containing memory profile of the model")
    
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="Micro-batch size per mini-batch")

    # need a better way to pass this information ?
    parser.add_argument("--total_num_stages", required=True, type=int,
                        help="The total number of potential stages/partitions the model is divided into")
    
    parser.add_argument("--gpus_per_stage", type=int, default = "0",
                        help="GPUs per stage (Only needed when we want to use less than ngpus_per_server * nservers)")

    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # parser.add_argument("placement", default='scattered', type=str,
    #                     help="Scattered/Clustered placement")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()
        

if __name__ == "__main__":

    print("Parent process ID:",os.getpid())

    with open("parent_process","w") as f:
        f.write(str(os.getpid()))

    args = parse_args()

    max_micro_batch_size_per_gpu = -1
    cutpoints_per_stage = args.total_num_stages // args.nstages

    with open(args.profile, 'r') as f:
        # skip line with column names
        f.readline()
        for line in f:
            if line == "":
                continue
            batch_size, _f, _b, max_mem_usage, _i , _m, _acts_size, _ = line.split(",")
            # profile is for single stage, so scale acc
            max_mem_usage = int(max_mem_usage) * cutpoints_per_stage
            batch_size = int(batch_size)
            if max_mem_usage > MAX_GPU_MEM:
                break
            else:
                max_micro_batch_size_per_gpu = batch_size

    print("max_micro_batch_size_per_gpu:",max_micro_batch_size_per_gpu)

    if max_micro_batch_size_per_gpu <= 0:
        raise ValueError("No micro-batch can fit for the model! Calculated max micro BS per gpu is " + str(max_micro_batch_size_per_gpu))

    # for max GPU util
    if args.chunk_size == -1:
        args.chunk_size = max_micro_batch_size_per_gpu

    with open('ngpus', 'w') as f:
        f.write(str(args.ngpus_per_server))
    with open('nservers', 'w') as f:
        f.write(str(args.nservers))

    def handler(signum,_):
        global loop_pending
        print('Signal handler called with signal', signum)
        # with open('ngpus','r') as f:
        #     ngpus_per_server = int(f.read())
        # with open('nservers','r') as f:
        #     nservers = int(f.read())
        # if args.ngpus_per_server == ngpus_per_server and args.nservers == nservers:
        #     return
        # loop_pending = True
        # if args.node_rank >= nservers:
        #     loop_pending = False
        # args.ngpus_per_server = ngpus_per_server
        # args.nservers = nservers
        for p in processes:
            p.send_signal(signal.SIGUSR1)
        # print("\n\n CONFIG CHANGED TO ",args.ngpus_per_server,"GPUS, ",args.nservers, "SERVERS","\n\n\n")
        print("\n\n CONFIG CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n\n\n")

    signal.signal(signal.SIGUSR1, handler)

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)

    if 'OMP_NUM_THREADS' not in os.environ and args.ngpus_per_server > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    loop_count = 0

    while True:
        processes = []
        loop_pending = False

        dist_world_size, stage_to_rank_map_str, ranks_in_server, train_batch_size = calculate_config(args)
        current_env["WORLD_SIZE"] = str(dist_world_size)
        print("World size is",dist_world_size)
        
        # uneven data parallelism not supported yet
        if dist_world_size % args.nstages != 0:
            raise ValueError("Each stage must get equal number of GPU processes")

        for rank in ranks_in_server:
            
            local_rank = rank % args.ngpus_per_server

            # each process's rank
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = [sys.executable, "-u"]
            # cmd = ["bash"]

            cmd.append(args.training_script)

            cmd.append("--rank={}".format(str(rank)))
            cmd.append("--partitions={}".format(str(args.nstages)))
            cmd.append("--chunk_size={}".format(str(args.chunk_size)))
            cmd.append("--local_rank={}".format(str(local_rank)))
            cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
            cmd.append("--device={}".format(str(local_rank_to_device[local_rank])))
            cmd.append("--train_batch_size={}".format(str(train_batch_size)))
            if loop_count > 0:
                cmd.append("--resume_from_checkpoint")

            cmd.extend(args.training_script_args)
            print(" ".join(cmd))

            # print(current_env["WORLD_SIZE"], current_env["RANK"], current_env["LOCAL_RANK"])
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        # cleanup on kill or error
        # def cleanup:

        # wait for all processes
        for process in processes:
            process.wait()
            print("Process done with return code", process.returncode)
            if process.returncode != 0:
                for p in processes:
                    p.kill()
                raise subprocess.CalledProcessError(returncode=process.returncode,
                                                    cmd=cmd)

        loop_count += 1

        if not loop_pending:
            print("Finished training!!")
            break

