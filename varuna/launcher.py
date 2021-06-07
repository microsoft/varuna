# Varuna launcher adapted from torch.distributed.launcher

import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
import math
import random
import socket

from .checkpoint import get_local_ckpt_tracker
from .utils import update_local_varuna_pid, VARUNA_TEMP_FOLDER, MORPH_PORT_ENV_VAR, HEARTBEAT_IP_ENV_VAR
from .auto_config import AutoConfig

processes = []

def calculate_config(args):
    # world size in terms of number of processes
    gpus_available = args.ngpus_per_server * args.nservers
    if args.nstages is None:
        args.nstages, args.chunk_size = num_partitions(gpus_available, args.ngpus_per_server, args.batch_size)
    gpus_per_stage = (gpus_available // args.nstages) if args.gpus_per_stage == 0 else args.gpus_per_stage
    # args.gpus_per_stage = gpus_per_stage
    print(gpus_per_stage, "per stage")
    dist_world_size = gpus_per_stage * args.nstages
    assert dist_world_size <= gpus_available, "Too many gpus_per_stage - {}!".format(gpus_per_stage)

    # some servers unused
    args.nservers = math.ceil(dist_world_size / float(args.ngpus_per_server))

    print(args.nservers, "servers!")
    if args.node_rank >= args.nservers:
        print(args.node_rank, args.nservers, "I am of no use!")
        exit()
    gpus_available = args.nservers * args.ngpus_per_server

    stage_to_rank_map = {}
    rank_to_stage_map = {}

    # clustered
    for i in range(args.nstages):
        stage_to_rank_map[i] = range(i, dist_world_size, args.nstages) 
        for r in stage_to_rank_map[i]:
            rank_to_stage_map[r] = i

    # scattered
    # for i in range(0,dist_world_size,gpus_per_stage):
    #    stage_to_rank_map[int(i//gpus_per_stage)] = range(i,i+gpus_per_stage)


    # # batch size should be divisible by num of data parallel workers
    per_gpu_batch_size = args.batch_size // gpus_per_stage
    total_batch_size = per_gpu_batch_size * gpus_per_stage     

    last_unused_gpus = 0
    if (dist_world_size % args.ngpus_per_server) != 0:
        last_unused_gpus = args.ngpus_per_server - (dist_world_size % args.ngpus_per_server)
    
    first_rank_in_server = args.node_rank * args.ngpus_per_server
    ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)
    if args.node_rank == args.nservers - 1:
        ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server - last_unused_gpus)

    stage_to_rank_map_str = ""
    for stage in stage_to_rank_map:
        ranks = ",".join([str(r) for r in stage_to_rank_map[stage]])
        stage_to_rank_map_str += (ranks + ";")

    print("Config:")
    print("ranks:", ranks_in_server)
    print("train batch size:",args.batch_size)
    print("partitions:", args.nstages)
    print("chunk_size:", args.chunk_size)
    print("data depth:", gpus_per_stage)
    print("stage to rank map:", stage_to_rank_map_str)

    return dist_world_size, stage_to_rank_map, ranks_in_server, total_batch_size, gpus_per_stage
    
def num_partitions(world_size, ngpus_per_server, batch_size):
    auto = AutoConfig(world_size, ngpus_per_server, batch_size)
    num_partitions, chunk_size, time = auto.get_min()
    print("best config is:", num_partitions, chunk_size)
    print("expected time is", time, flush=True)
    return num_partitions, chunk_size

def send_to_manager(message, manager_ip, manager_port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((manager_ip, manager_port))
            sock.sendall(bytes(message, 'ascii'))
    except Exception as e:
        print(f"Could not send message: {message}", e)

def get_last_iter(num_local_processes):
    last_iter = -1
    for i in range(num_local_processes):
        ckpt_tracker = get_local_ckpt_tracker(i)
        if os.path.exists(ckpt_tracker):
            with open(ckpt_tracker,"r") as f:
                last_iter_ = int(f.read())
        else:
            last_iter_ = -1

        if last_iter == -1:
            last_iter = last_iter_
        else:
            last_iter = min(last_iter, last_iter_) 
    return last_iter

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Varuna training launch "
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
    parser.add_argument("--chunk_size", type=int, required=True,
                        help="Micro-batch size per mini-batch")
    parser.add_argument("--code_dir", default=None, type=str,
                        help="Directory to run training in")
    parser.add_argument("--gpus_per_stage", type=int, default = "0",
                        help="GPUs per stage (Only needed when we want to use less than ngpus_per_server * nservers)")
    # need a better way to pass this information ?
    # parser.add_argument("--total_num_stages", required=True, type=int,
    #                     help="The total number of potential stages/partitions the model is divided into")
    
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument("--custom_placement", default=False, action="store_true",
                        help="place embeddings separately if possible")

    # parser.add_argument("--rank_aliasing", default=False, action="store_true",
    #                     help="shuffle ranks to avoid NCCL errors")

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

    if not os.path.exists(VARUNA_TEMP_FOLDER):
        os.makedirs(VARUNA_TEMP_FOLDER)

    update_local_varuna_pid(os.getpid())

    args = parse_args()
    manager_ip = os.environ[HEARTBEAT_IP_ENV_VAR]
    manager_port = int(os.environ[MORPH_PORT_ENV_VAR])

    def handler(signum,_):
        global loop_pending
        print('Signal handler called with signal', signum, flush=True)
        loop_pending = False
        try:
            for p in processes:
                p.send_signal(signal.SIGUSR1)
        except Exception as e:
            print("run_varuna: error while sending signal:- ", e)
            
        print("\n\n STOPPING VARUNA !!\n\n\n", flush=True)

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

    dist_world_size, stage_to_rank_map, ranks_in_server, \
        total_batch_size, gpus_per_stage = calculate_config(args)

    alias_ranks = list(range(dist_world_size))

    if args.node_rank == 0:
        send_to_manager("starting job of size {}".format(dist_world_size), manager_ip, manager_port)

    current_env["WORLD_SIZE"] = str(dist_world_size)
    print("World size is",dist_world_size)
    
    # uneven data parallelism not supported yet
    if dist_world_size % args.nstages != 0:
        raise ValueError("Each stage must get equal number of GPU processes")

    stage_to_rank_map_str = ""
    for stage in stage_to_rank_map:
        ranks = ",".join([str(r) for r in stage_to_rank_map[stage]])
        stage_to_rank_map_str += (ranks + ";")

    for rank in ranks_in_server:
        
        local_rank = rank % args.ngpus_per_server
        rank = alias_ranks[rank]            

        # each process's rank
        current_env["RANK"] = str(rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, "-u"]
        cmd.append(args.training_script)

        per_process_batch_size = total_batch_size // gpus_per_stage

        cmd.append("--rank={}".format(str(rank)))
        cmd.append("--chunk_size={}".format(str(args.chunk_size)))
        cmd.append("--local_rank={}".format(str(local_rank)))
        cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
        cmd.append("--batch-size={}".format(str(per_process_batch_size)))

        cmd.extend(args.training_script_args)
        print(" ".join(cmd), flush=True)

        process = subprocess.Popen(cmd, env=current_env,cwd=args.code_dir)
        processes.append(process)

    # wait for all processes
    try:
        for process in processes:
            process.wait()
            print("Process done with return code", process.returncode)
            if process.returncode != 0:
                for p in processes:
                    p.kill()
    except Exception as e:
        print("run_varuna subprocesses quit with error:", e)


    last_iter = get_last_iter(len(ranks_in_server))
    send_to_manager("checkpoint done {}".format(last_iter), manager_ip, manager_port)

