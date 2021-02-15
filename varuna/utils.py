
import os
import socket
import math

try:
    import torch
    from apex import amp
    from apex.amp import _amp_state
except:
    pass
    
VARUNA_TEMP_FOLDER = "/tmp/varuna"
HEARTBEAT_IP_ENV_VAR = "VARUNA_MANAGER_IP"
HEARTBEAT_PORT_ENV_VAR = "VARUNA_HEARTBEAT_PORT"
MORPH_PORT_ENV_VAR = "VARUNA_MORPH_PORT"
LOCAL_PID_FILENAME = "local_parent_pid"

def scatter(input, batch_size, chunk_size):
    """
    Accepts input dictionary and splits into microbatches
    """
    assert isinstance(input,dict) , "varuna inputs must be given as a dictionary" 
    
    microbatches = []
    num_microbatches = math.ceil(batch_size / chunk_size)
    for k,v in input.items():
        # TODO: what will happen for indivisibilities in uneven data parallelism !!
        # print(dist.get_rank(),k,v.size())
        # special case for GPT-2 attention mask
        if v.size(0) == 1:
            chunked_values = [v for _ in range(num_microbatches)]
        else:
            chunked_values = v.split(chunk_size)
        for i,value in enumerate(chunked_values):
            if len(microbatches) <= i:
                microbatches.append(dict())
            microbatches[i][k]=value
    
    return microbatches

def save_rng_states(device):
    """capture current CPU, GPU random number generator states to reuse while recomputing activations
    in order to ensure Referential Transparency
    """
    cpu_rng_state = torch.get_rng_state()

    gpu_rng_states: Optional[ByteTensor]
    # gpu_rng_states = torch.cuda.get_rng_state_all() 
    gpu_rng_states = torch.cuda.get_rng_state(device)
    return (cpu_rng_state, gpu_rng_states)

def restore_rng_states(rng_states, device):
    cpu_rng_state, gpu_rng_states = rng_states
    torch.set_rng_state(cpu_rng_state)
    # torch.cuda.set_rng_state_all(gpu_rng_states)        # todo: verify correctness;   batchNorm, dropouts, convlayers?
    torch.cuda.set_rng_state(gpu_rng_states, device)


def clip_grad_norm(parameters, total_norm, max_norm):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    modified to handle pipeline parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    
    # print(f'clip_grad_norm() total_norm = {total_norm}')
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
            
    return clip_coef<1

def heartbeat(step, ip, port):
    if (ip is not None) and (port is not None):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                message = "progress {}".format(step)
                sock.connect((ip, port))
                sock.sendall(bytes(message, 'ascii'))
        except:
            print("Could not send progress update message")

def generate_schedule(chunks, stage, partitions):
    print(chunks,"chunks")
    gensched_binary = os.path.join(os.path.dirname(os.path.abspath(__file__)),'genschedule')
    c_schedule = os.popen( gensched_binary + ' ' +
                            str(partitions) + ' ' +
                            str(chunks) + ' ' +
                            str(stage)).read()
    schedule = list()
    steps = c_schedule.split(';')
    steps = steps[:-1]
    for step in steps:
        task = step.split(',')
        schedule.append((int(task[0]), int(task[1])))
    print("schedule",schedule)
    return schedule

def parse_stage_to_rank_map(stage_to_rank_map_str):
    """ parses the stage_to_rank_map string recieved from varuna launcher """
    # parse stage_to_rank_map
    stage_ranks = stage_to_rank_map_str.split(";")[:-1]
    partitions = len(stage_ranks)
    assert partitions > 0, "Invalid stage to rank map for Varuna!"
    stage_to_rank_map = {}
    for i in range(partitions):
        ranks = stage_ranks[i].split(",")
        stage_to_rank_map[int(i)] = [int(r) for r in ranks]
    return stage_to_rank_map

def get_varuna_config(stage_to_rank_map_str):
    """ parses the stage_to_rank_map string recieved from varuna launcher to
        return a tuple of the form (num_pipeline_stages, num_data_parallel_replicas)"""
    stage_to_rank_map = parse_stage_to_rank_map(stage_to_rank_map_str)
    return len(stage_to_rank_map), len(stage_to_rank_map[0])

def get_this_rank_config_varuna(stage_to_rank_map_str, rank):
    """ parses the varuna stage_to_rank_map string and for a given rank
        returns a tuple of the form (my_stage, my_data_parallel_rank)"""
    stage_to_rank_map = parse_stage_to_rank_map(stage_to_rank_map_str)
    for stage in stage_to_rank_map:
        if rank in stage_to_rank_map[stage]:
            my_stage = stage
            my_dp_rank = stage_to_rank_map[stage].index(rank)
            return my_stage, my_dp_rank
    raise RuntimeError(f"rank {rank} not present in varuna config!!")

def is_varuna_dummy_val(val):
    if isinstance(val,tuple) and len(val)>0:
        val = val[0]
        return hasattr(val, "varuna_valid") and not val.varuna_valid
    return (val is None)

def get_heartbeat_server_info():
    ip = os.environ.get(HEARTBEAT_IP_ENV_VAR, None)
    port = os.environ.get(HEARTBEAT_PORT_ENV_VAR, None)
    return ip, port

def update_local_varuna_pid(pid):
    with open(os.path.join(VARUNA_TEMP_FOLDER, LOCAL_PID_FILENAME), "w") as f:
        f.write(str(pid))

def get_local_varuna_pid():
    with open(os.path.join(VARUNA_TEMP_FOLDER, LOCAL_PID_FILENAME), "w") as f:
        pid = int(f.read().strip())
    return pid