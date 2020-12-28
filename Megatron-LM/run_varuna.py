import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
import math
import random
import socket
# import atexit

local_rank_to_device = [0,1,2,3]

# different for diff kinds of GPUs
MAX_GPU_MEM = 16280000000
processes = []

manager_ip = "10.0.3.4"
manager_port = 4201

def calculate_config(args):
    # world size in terms of number of processes
    gpus_available = args.ngpus_per_server * args.nservers
    gpus_per_stage = (gpus_available // args.nstages) if args.gpus_per_stage == 0 else args.gpus_per_stage
    # args.gpus_per_stage = gpus_per_stage
    print(gpus_per_stage, "per stage")
    dist_world_size = gpus_per_stage * args.nstages
    assert dist_world_size <= gpus_available, "Too many gpus_per_stage - {}!".format(gpus_per_stage)

    # some servers unused
    if args.custom_placement:
        servers_for_embeddings = math.ceil(gpus_per_stage / float(args.ngpus_per_server))
        other_servers = math.ceil((dist_world_size - gpus_per_stage) / float(args.ngpus_per_server))
        num_servers = other_servers + servers_for_embeddings
        if num_servers > args.nservers:
            raise RuntimeError("Not enough servers for cutom placement")
        args.nservers = num_servers
    else:
        args.nservers = math.ceil(dist_world_size / float(args.ngpus_per_server))

    print(args.nservers, "servers!")
    if args.node_rank >= args.nservers:
        print(args.node_rank, args.nservers, "I am of no use!")
        exit()
    gpus_available = args.nservers * args.ngpus_per_server

    stage_to_rank_map = {}
    rank_to_stage_map = {}

    if args.custom_placement:
        # seperate VMs for embeddings
        stage_to_rank_map[0] = range(0, gpus_per_stage)
        for i in range(1, args.nstages):
            stage_to_rank_map[i] = \
                range( gpus_per_stage + i-1, dist_world_size, args.nstages-1)
    else:
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

    if args.custom_placement:
        last_unused_gpus = 0
        if ((dist_world_size - gpus_per_stage) % args.ngpus_per_server) != 0:
            last_unused_gpus = args.ngpus_per_server - ((dist_world_size - gpus_per_stage) % args.ngpus_per_server)

        last_embedding_node_rank = math.ceil(gpus_per_stage / args.ngpus_per_server) - 1
        # unused_gpus = (args.nservers - last_embedding_node_rank - 1) * args.ngpus_per_server
        if args.node_rank == last_embedding_node_rank:
            first_rank_in_server = args.node_rank * args.ngpus_per_server
            ranks_in_server = range(first_rank_in_server, gpus_per_stage)
        elif args.node_rank < last_embedding_node_rank:
            first_rank_in_server = args.node_rank * args.ngpus_per_server
            ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)
        elif args.node_rank < (args.nservers - 1):
            first_rank_in_server = (args.node_rank * args.ngpus_per_server) - (args.ngpus_per_server - gpus_per_stage % args.ngpus_per_server)
            ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)
        else:
            first_rank_in_server = (args.node_rank * args.ngpus_per_server) - (args.ngpus_per_server - gpus_per_stage % args.ngpus_per_server)
            ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server - last_unused_gpus)
    
    else:
        last_unused_gpus = 0
        if (dist_world_size % args.ngpus_per_server) != 0:
            last_unused_gpus = args.ngpus_per_server - (dist_world_size % args.ngpus_per_server)
        
        first_rank_in_server = args.node_rank * args.ngpus_per_server
        ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)
        if args.node_rank == args.nservers - 1:
            ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server - last_unused_gpus)
    

    # shuffle ranks for some razzle dazzle
    alias_map_str = None
    if args.rank_aliasing:
        all_ranks = list(range(dist_world_size))
        random.Random(4).shuffle(all_ranks)

        all_reduce_groups = [all_ranks[i:i+gpus_per_stage] for i in range(0,dist_world_size, gpus_per_stage)]
        all_reduce_groups = [sorted(g) for g in all_reduce_groups]

        alias_map = []
        node_rank0 = -1
        for r in range(dist_world_size):
            replica_num = r // args.nstages
            s = rank_to_stage_map[r]
            agr = (replica_num + s) % gpus_per_stage
            alias = all_reduce_groups[s][agr]
            assert alias not in alias_map, "Whhaaaa"+str(r)
            if alias == 0:
                node_rank0 = r // args.ngpus_per_server
            alias_map.append(alias)


        print("all ranks", len(alias_map))
        all_ranks_str = [str(r) for r in alias_map]
        alias_map_str = ",".join(all_ranks_str)
        for s in stage_to_rank_map:
            orig_ranks = stage_to_rank_map[s]
            alias_ranks = [alias_map[r] for r in orig_ranks]
            stage_to_rank_map[s] = alias_ranks

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

    return dist_world_size, stage_to_rank_map, ranks_in_server, total_batch_size, gpus_per_stage, alias_map_str
    

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
    
    parser.add_argument("--chunk_size", type=int, required=True,
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
    parser.add_argument("--custom_placement", default=False, action="store_true",
                        help="place embeddings separately if possible")

    parser.add_argument("--rank_aliasing", default=False, action="store_true",
                        help="shuffle ranks to avoid NCCL errors")

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

    def handler(signum,_):
        global loop_pending
        print('Signal handler called with signal', signum, flush=True)
        with open('ngpus','r') as f:
            ngpus_per_server = int(f.read())
        with open('nservers','r') as f:
            nservers = int(f.read())
        with open('nstages','r') as f:
            nstages = int(f.read())
        with open('gpus_per_stage','r') as f:
            gpus_per_stage = int(f.read())
        if args.ngpus_per_server == ngpus_per_server and args.nservers == nservers and args.nstages == nstages and args.gpus_per_stage == gpus_per_stage:
            return
        loop_pending = True
        if args.node_rank >= nservers:
            loop_pending = False
        args.ngpus_per_server = ngpus_per_server
        args.nservers = nservers
        args.gpus_per_stage = gpus_per_stage
        args.nstages = nstages
        try:
            for p in processes:
                p.send_signal(signal.SIGUSR1)
        except Exception as e:
            print("run_varuna: error while sending signal:- ", e)
            
        print("\n\n CONFIG CHANGED TO ",args.nservers, "x",args.ngpus_per_server,"!!\n\n\n", flush=True)

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

        dist_world_size, stage_to_rank_map, ranks_in_server, \
            total_batch_size, gpus_per_stage, alias_map = calculate_config(args)

        if alias_map is not None:
            alias_ranks = [int(i) for i in alias_map.split(",")]
        else:
            alias_ranks = list(range(dist_world_size))

        print("ALIAS RANKS:", alias_ranks)

        
        if args.node_rank == 0:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    message = "starting job of size {}".format(dist_world_size)
                    sock.connect((manager_ip, manager_port))
                    sock.sendall(bytes(message, 'ascii'))
            except:
                print("Could not send start message")
        

        with open('ngpus', 'w') as f:
            f.write(str(args.ngpus_per_server))
        with open('nservers', 'w') as f:
            f.write(str(args.nservers))
        with open('nstages', 'w') as f:
            f.write(str(args.nstages))
        with open('gpus_per_stage', 'w') as f:
            f.write(str(args.gpus_per_stage))

        current_env["WORLD_SIZE"] = str(dist_world_size)
        print("World size is",dist_world_size)
        
        # uneven data parallelism not supported yet
        if dist_world_size % args.nstages != 0:
            raise ValueError("Each stage must get equal number of GPU processes")

        if args.rank_aliasing:
            group_ranks = []
            for rr in ranks_in_server:
                r = alias_ranks[rr]
                for s in stage_to_rank_map:
                    if r in stage_to_rank_map[s]:
                        gr = sorted(stage_to_rank_map[s]).index(r)
                        group_ranks.append(gr)
                        break
            collision = 0
            for i in range(4):
                for j in range(i+1,4):
                    if group_ranks[i] == group_ranks[j]:
                        collision += 1
            print(group_ranks)
            print("Collisions",collision)

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
            cmd.append("--partitions={}".format(str(args.nstages)))
            cmd.append("--chunk_size={}".format(str(args.chunk_size)))
            cmd.append("--local_rank={}".format(str(local_rank)))
            cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
            cmd.append("--batch-size={}".format(str(per_process_batch_size)))

            cmd.extend(args.training_script_args)
            print(" ".join(cmd), flush=True)

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        # wait for all processes
        try:
            for process in processes:
                process.wait()
                print("Process done with return code", process.returncode)
                if process.returncode != 0:
                    for p in processes:
                        p.kill()
                    # raise subprocess.CalledProcessError(returncode=process.returncode,
                                                        # cmd=cmd)
        except Exception as e:
            print("run_varuna subprocesses quit with error:", e)


        last_iter = -1
        if os.path.exists("/home/varuna/local_ckpt_tracker.txt"):
            with open("/home/varuna/local_ckpt_tracker.txt","r") as f:
                last_iter = int(f.read())

        print("done and trying to send here", flush=True)
     
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                message = "checkpoint done {}".format(last_iter)
                sock.connect((manager_ip, manager_port))
                sock.sendall(bytes(message, 'ascii'))
        except:
            print('Could not send checkpoint done signal')

        print("and sent!")

        loop_count += 1

        if not loop_pending:
            print("Finished training!!")
            break

