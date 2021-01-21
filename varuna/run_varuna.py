
import os, subprocess
from argparse import ArgumentParser, REMAINDER

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Varuna training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")
    parser.add_argument("--machine_list", type=str, default="available_machines.out",
                            help = "path to a file with reachable IPs written line-wise." 
                            "These should be available to ssh into through the manager ")
    parser.add_argument("--gpus_per_node", type=int, default=4,
                            help = "number of GPUs per machine")

    # launch worker args
    parser.add_argument("--nstages", type=int, required = True,
                        help="Depth of pipeline (number of stages)")
    parser.add_argument("--batch_size", required=True, type=int,
                        help="Total effective batch size for training")
    parser.add_argument("--chunk_size", type=int, required=True,
                        help="Micro-batch size per mini-batch")
    parser.add_argument("--code_dir", default=None, type=str,
                        help="Directory to run training in")

    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(args.machine_list, "r") as f:
        reachable_machines = f.read().split("\n")
        reachable_machines = [m for m in reachable_machines if len(m) > 0]  
    print(reachable_machines)  
    
    reachable_count = len(reachable_machines)
    if reachable_count == 0:
        print("Empty machine list, nothing to run!")
        exit()

    if args.code_dir is None:
        args.code_dir = os.getcwd()

    master_addr = reachable_machines[0]
    for i,machine in enumerate(reachable_machines):
        launch_cmd = []
        launch_cmd.append(f"/home/varuna/anaconda3/bin/python -u -m varuna.launcher --nservers {reachable_count}" \
            +  f" --ngpus_per_server {args.gpus_per_node} --node_rank {i} --master_addr {master_addr} " \
            +  f" --nstages {args.nstages} --batch_size {args.batch_size}" \
            +  f" --chunk_size {args.chunk_size} --code_dir {args.code_dir}")
        #  $i $master_addr $ckpt"
        launch_cmd.append(args.training_script)
        launch_cmd.extend(args.training_script_args)
        # launch_cmd.append("\"")
        launch_cmd = " ".join(launch_cmd)
        cmd = ["ssh"]
        # cmd.append("-i /home/varuna/.ssh/vdummy.pem")
        cmd.append(machine)
        cmd.append(f"echo \"{launch_cmd}\" > launch_varuna.sh; bash launch_varuna.sh")
        current_env = os.environ.copy()
        print(" ".join(cmd ))
        out_file = open(f"ssh_logs/my_ssh_out_{i}","w")
        err_file = open(f"ssh_logs/my_ssh_err_{i}", "w")
        process = subprocess.Popen(cmd, env=current_env, 
                                    stdout=out_file,
                                    stderr=err_file)







