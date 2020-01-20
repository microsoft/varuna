import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
import math
# import atexit

loop_pending = False


# different for diff kinds of GPUs
MAX_GPU_MEM = 16280000000

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

    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    parser.add_argument("--batch_size", required=True, type=int,
                        help="Total effective batch size required")

    parser.add_argument("--chunks", required=True, type=int,
                        help="Number of micro-batches per mini-batch")

    parser.add_argument("--profile", required=True, type=str,
                        help="CSV file containing memory profile of the model")

    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()
        

if __name__ == "__main__":

    print("Parent process ID:",os.getpid())

    with open("parent_process","w") as f:
        f.write(str(os.getpid()))

    args = parse_args()

    max_micro_batch_size_per_gpu = -1
    cutpoints_per_stage = 24 // args.nstages
    max_mem = {}
    with open(args.profile, 'r') as f:
        # skip line with column names
        f.readline()
        for line in f:
            if line == "":
                continue
            batch_size, _f, _b, max_mem_usage, _i , _m, _ = line.split(",")
            # profile is for single stage, so scale acc
            max_mem_usage = int(max_mem_usage) * cutpoints_per_stage
            batch_size = int(batch_size)
            if max_mem_usage > MAX_GPU_MEM:
                max_micro_batch_size_per_gpu = batch_size - 1
                break
            max_mem[batch_size] = max_mem_usage

    print("max_micro_batch_size_per_gpu:",max_micro_batch_size_per_gpu)

    # if max_micro_batch_size_per_gpu <= 0:
    #     ERRORR

    with open('ngpus', 'w') as f:
        f.write(str(args.ngpus_per_server))
    with open('nservers', 'w') as f:
        f.write(str(args.nservers))


    def handler(signum,_):
        global loop_pending
        print('Signal handler called with signal', signum)
        loop_pending = True
        with open('ngpus','r') as f:
            ngpus_per_server = int(f.read())
        with open('nservers','r') as f:
            nservers = int(f.read())
        if args.ngpus_per_server == ngpus_per_server and args.nservers == nservers:
            return
        if args.node_rank >= nservers:
            loop_pending = False
        args.ngpus_per_server = ngpus_per_server
        args.nservers = nservers
        for p in processes:
            p.send_signal(signal.SIGUSR1)
        # send signal to other nodes as well
        # if args.node_rank == 0:
        print("\n\n==========================  NUM OF WORKERS CHANGED TO ",args.ngpus_per_server," ===========================\n\n\n")

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

    count = 0

    while True:
        processes = []
        loop_pending = False
        
        # world size in terms of number of processes
        dist_world_size = args.ngpus_per_server * args.nservers
        current_env["WORLD_SIZE"] = str(dist_world_size)
        print("World size is",dist_world_size)

        # uneven data parallelism not supported yet
        # if dist_world_size % args.nstages != 0:
        #     raise ValueError("Each stage must get equal number of GPU processes")
        gpus_per_stage = dist_world_size // args.nstages
        # print("WARNING: ", str(dist_world_size % args.nstages), "gpus going unused!!")
    
        first_rank_in_server = args.node_rank * args.ngpus_per_server
        ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)

        stage_to_rank_map = {}
        rank_to_stage_map = {}

        for i in range(dist_world_size):
            stage = i // gpus_per_stage
            if stage not in stage_to_rank_map:
                stage_to_rank_map[stage] = []
            stage_to_rank_map[stage].append(i)
            rank_to_stage_map[i] = stage

        stage_to_rank_map_str = ""
        for stage in stage_to_rank_map:
            ranks = ",".join([str(r) for r in stage_to_rank_map[stage]])
            stage_to_rank_map_str += (ranks + ";")

        per_gpu_batch_size = args.batch_size // gpus_per_stage
        per_gpu_micro_batch_size = math.ceil(per_gpu_batch_size / (1.0 * args.chunks))
        train_batch_size = args.batch_size
        gradient_accumulation_steps = 1
        max_gradient_accumulation_steps = per_gpu_micro_batch_size // max_micro_batch_size_per_gpu

        while per_gpu_micro_batch_size > max_micro_batch_size_per_gpu and gradient_accumulation_steps <= max_gradient_accumulation_steps :
            gradient_accumulation_steps += 1
            per_gpu_micro_batch_size_ = per_gpu_micro_batch_size // gradient_accumulation_steps 
            if per_gpu_micro_batch_size_ <= max_micro_batch_size_per_gpu:
                per_gpu_micro_batch_size = per_gpu_micro_batch_size_
                train_batch_size = per_gpu_micro_batch_size * args.chunks * gpus_per_stage
                break

        print("Config:")
        print("grad acc steps:", gradient_accumulation_steps)
        print("train batch size:",train_batch_size)
        print("partitions:", args.nstages)
        print("data depth:", gpus_per_stage)
        print("stage to rank map:", stage_to_rank_map_str)

        for rank in ranks_in_server:
            
            local_rank = rank % args.ngpus_per_server

            # each process's rank
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = [sys.executable, "-u"]

            cmd.append(args.training_script)

            cmd.append("--rank={}".format(rank))
            cmd.append("--local_rank={}".format(local_rank))
            cmd.append("--partitions={}".format(args.nstages))
            cmd.append("--chunks={}".format(args.chunks))
            cmd.append("--train_batch_size={}".format(train_batch_size))
            cmd.append("--gradient_accumulation_steps={}".format(gradient_accumulation_steps))
            cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
            if count > 0:
                cmd.append("--resume")
            cmd.extend(args.training_script_args)

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

        count += 1

        if not loop_pending:
            print("Finished training!!")
            break