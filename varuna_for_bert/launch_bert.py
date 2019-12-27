import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
# import atexit

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
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.ngpus_per_server * args.nservers
    
    # uneven data parallelism not supported yet
    if dist_world_size % args.nstages != 0:
        raise ValueError("Each stage must get equal number of GPU processes")
    
    gpus_per_stage = dist_world_size / args.nstages
    
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
        
    # stage_to_rank_map_str = "0,1,2;3,4,5;6,7"
    # print("Map is",stage_to_rank_map_str)


    first_rank_in_server = args.node_rank * args.ngpus_per_server
    ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    if 'OMP_NUM_THREADS' not in os.environ and args.ngpus_per_server > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    processes = []

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
        # cmd.append("--stage={}".format(stage))
        # cmd.append("--first_rank_in_server={}".format(first_rank_in_server))
        # cmd.append("--ngpus_per_server={}".format(args.ngpus_per_server))
        cmd.append("--partitions={}".format(args.nstages))
        cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
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
