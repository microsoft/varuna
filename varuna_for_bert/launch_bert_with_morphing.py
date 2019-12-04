import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
# import atexit

loop_pending = False
max_batch_size_per_gpu = 630

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

    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    parser.add_argument("--n_stages", default=8, type=int,
                        help="Total number os stages/partitions set initially")

    parser.add_argument("--batch_size", required=True, type=int,
                        help="Total effective batch size required")

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
    args.n_stages = args.ngpus_per_server * args.nservers
    dist_world_size = args.n_stages

    def handler(signum,_):
        global loop_pending
        print('Signal handler called with signal', signum)
        loop_pending = True
        with open('n_stages','r') as f:
            n_stages = int(f.read())
        if args.n_stages == n_stages:
            return
        args.n_stages = n_stages
        for p in processes:
            p.send_signal(signal.SIGUSR1)
        print("\n\n==========================  NUM OF PARTITIONS CHANGED TO ",args.n_stages," ===========================\n\n\n")

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

    processes = []
    count = 0

    while True:

        first_rank_in_server = args.node_rank * args.n_stages
        # ranks_in_server = range(first_rank_in_server, first_rank_in_server + args.ngpus_per_server)
        ranks_in_server = range(args.n_stages)

        stage_to_rank_map = {}
        rank_to_stage_map = {}

        for i in ranks_in_server:
            stage_to_rank_map[i] = [i]
            rank_to_stage_map[i] = i 

        dist_world_size = args.n_stages

        per_gpu_train_batch_size = args.batch_size
        gradient_accumulation_steps = 1

        if per_gpu_train_batch_size > max_batch_size_per_gpu:
            gradient_accumulation_steps = per_gpu_train_batch_size // max_batch_size_per_gpu
            per_gpu_train_batch_size = max_batch_size_per_gpu

        current_env["WORLD_SIZE"] = str(dist_world_size)

        print("World size is",dist_world_size)

        for rank in ranks_in_server:
            
            stage = rank_to_stage_map[rank]
            local_rank = rank % args.ngpus_per_server

            # each process's rank
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(local_rank)
            current_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

            # spawn the processes
            cmd = [sys.executable, "-u"]

            cmd.append(args.training_script)

            cmd.append("--rank={}".format(rank))
            cmd.append("--local_rank={}".format(local_rank))
            cmd.append("--partitions={}".format(args.n_stages))
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

        
