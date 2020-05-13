import os
from argparse import ArgumentParser, REMAINDER
import sys
import subprocess
import signal
import math
import socket
# import atexit

local_rank_to_device = [0,1,2,3]

# different for diff kinds of GPUs
MAX_GPU_MEM = 16280000000
processes = []

manager_ip = "172.16.5.4"
manager_port = 4200

def calculate_config(args):
    # world size in terms of number of processes
    gpus_available = args.ngpus_per_server * args.nservers
    gpus_per_stage = (gpus_available // args.nstages) if args.gpus_per_stage == 0 else args.gpus_per_stage
    # args.gpus_per_stage = gpus_per_stage
    print(gpus_per_stage, "per stage")
    dist_world_size = gpus_per_stage * args.nstages
    assert dist_world_size <= gpus_available, "Too many gpus_per_stage - {}!".format(gpus_per_stage)

    # some servers unused
    # if unused_gpus > args.ngpus_per_server:
    #     raise ValueError("Wrong number of servers - too many unused GPUs")
    num_servers = math.ceil(dist_world_size / args.ngpus_per_server)
    if args.node_rank >= num_servers:
        print(args.node_rank, num_servers, "I am of no use!")
        exit()
    args.nservers = num_servers
    gpus_available = args.nservers * args.ngpus_per_server
    if (dist_world_size % args.ngpus_per_server) != 0:
        unused_gpus = args.ngpus_per_server - (dist_world_size % args.ngpus_per_server)
    else:
        unused_gpus = 0

    stage_to_rank_map = {}
    rank_to_stage_map = {}

    for i in range(args.nstages):
        stage_to_rank_map[i]= range(i,dist_world_size,args.nstages)
#    for i in range(0,dist_world_size,gpus_per_stage):
#        stage_to_rank_map[int(i//gpus_per_stage)] = range(i,i+gpus_per_stage)
    stage_to_rank_map_str = ""
    for stage in stage_to_rank_map:
        ranks = ",".join([str(r) for r in stage_to_rank_map[stage]])
        stage_to_rank_map_str += (ranks + ";")

    # # batch size should be divisible by num of data parallel workers
    per_gpu_batch_size = args.batch_size // gpus_per_stage
    total_batch_size = per_gpu_batch_size * gpus_per_stage
    # train_batch_size = args.batch_size
    
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

    return dist_world_size, stage_to_rank_map_str, ranks_in_server, total_batch_size, gpus_per_stage
    


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

    # parser.add_argument("--profile", required=True, type=str,
    #                     help="CSV file containing memory profile of the model")
    
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

    # max_micro_batch_size_per_gpu = -1
    # cutpoints_per_stage = args.total_num_stages // args.nstages
    # with open(args.profile, 'r') as f:
    #     # skip line with column names
    #     f.readline()
    #     for line in f:
    #         if line == "":
    #             continue
    #         batch_size, _f, _b, max_mem_usage, _i , _m, _acts_size, _ = line.split(",")
    #         # profile is for single stage, so scale acc
    #         max_mem_usage = int(max_mem_usage) * cutpoints_per_stage
    #         batch_size = int(batch_size)
    #         if max_mem_usage > MAX_GPU_MEM:
    #             break
    #         else:
    #             max_micro_batch_size_per_gpu = batch_size

    # print("max_micro_batch_size_per_gpu:",max_micro_batch_size_per_gpu)

    # if max_micro_batch_size_per_gpu <= 0:
    #     raise ValueError("No micro-batch can fit for the model! Calculated max micro BS per gpu is " + str(max_micro_batch_size_per_gpu))

    # for max GPU util
    if args.chunk_size == -1:
        args.chunk_size = max_micro_batch_size_per_gpu

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
        for p in processes:
            p.send_signal(signal.SIGUSR1)
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

        dist_world_size, stage_to_rank_map_str, ranks_in_server, total_batch_size, gpus_per_stage = calculate_config(args)
        
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

        for rank in ranks_in_server:
            
            local_rank = rank % args.ngpus_per_server

            # each process's rank
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = [sys.executable, "-u"]
            # cmd = ["bash"]

            cmd.append(args.training_script)

            per_process_batch_size = total_batch_size // gpus_per_stage

            cmd.append("--rank={}".format(str(rank)))
            cmd.append("--partitions={}".format(str(args.nstages)))
            cmd.append("--chunk_size={}".format(str(args.chunk_size)))
            cmd.append("--local_rank={}".format(str(local_rank)))
            cmd.append("--stage_to_rank_map={}".format(stage_to_rank_map_str))
            # cmd.append("--device={}".format(str(local_rank_to_device[local_rank])))
            cmd.append("--batch-size={}".format(str(per_process_batch_size)))
            # if loop_count > 0:
            #     with open("resume_step","r") as f:
            #         resume_step = int(f.read())
            #     cmd.append("--resume_from_checkpoint")
            #     cmd.append("--resume_step={}".format(resume_step))

            cmd.extend(args.training_script_args)
            print(" ".join(cmd), flush=True)

            # print(current_env["WORLD_SIZE"], current_env["RANK"], current_env["LOCAL_RANK"])
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        # cleanup on kill or error
        # def cleanup:

        with open("prev_job_done","w")as f:
            f.write("notdone")

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

        with open("prev_job_done","w") as f:
            f.write("done")

        last_iter = -1
        if os.path.exists("/home/varuna/local_ckpt_tracker.txt"):
            with open("/home/varuna/local_ckpt_tracker.txt","r") as f:
                last_iter = int(f.read())

        print("done and trying to send here", flush=True)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            message = "checkpoint done {}".format(last_iter)
            sock.connect((manager_ip, manager_port))
            sock.sendall(bytes(message, 'ascii'))

        print("and sent!")

        loop_count += 1

        if not loop_pending:
            print("Finished training!!")
            break

