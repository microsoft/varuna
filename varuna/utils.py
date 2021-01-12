
import os
import torch
import socket
import math

from apex import amp
from apex.amp import _amp_state

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


def clip_grad_norm(parameters, grad_norm_sq, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
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
    norm_type = float(norm_type)
    
    total_norm = grad_norm_sq.item() ** (1. / norm_type)
    # print(f'clip_grad_norm() total_norm = {total_norm}')
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
            
    return clip_coef<1


def load_varuna_checkpoint(my_stage, num_stages, total_num_pstages, common_store, \
                        prefix="cp-pstage", pstages_to_read = None):
    state_dict = {}
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    for i in pstages_to_read:
        cp_file = os.path.join(common_store, "{}-{}".format(prefix,i))
        if not os.path.exists(cp_file):
            print("WARNING: DID NOT FIND CKPT FILE",cp_file,"!!!!")
            continue
        state_dict_ = torch.load(cp_file,map_location="cpu")
        state_dict.update(state_dict_)
    return state_dict

def load_varuna_optimizer(optimizer, my_stage, num_stages, total_num_pstages, parameter_names, \
                        common_store, fp16=False, pstages_to_read = None):
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    # reload state
    opt_state = {}
    for i in pstages_to_read:
        state_ = torch.load(os.path.join(common_store,"opt-state-{}".format(i)),map_location='cpu')
        opt_state.update(state_)
    for p in amp.master_params(optimizer):
        name = parameter_names[p]
        if name in opt_state:
            optimizer.state[p] = opt_state[name]
    # reload master params
    if fp16:
        saved_master_params = dict()
        for i in pstages_to_read:
            params_ = torch.load(os.path.join(common_store, "opt-fp32-params-{}".format(i)),map_location="cpu")
            saved_master_params.update(params_)
        for p in amp.master_params(optimizer):
            name = parameter_names[p]
            if name in saved_master_params:
                p.data.copy_(saved_master_params[name].data)

def heartbeat(step):
    manager_ip = "10.0.3.4"
    manager_port = 5000
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            message = "progress {}".format(step)
            sock.connect((manager_ip, manager_port))
            sock.sendall(bytes(message, 'ascii'))
    except:
        print("Could not send progress update message")