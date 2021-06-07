
import os,sys, time
import torch
import concurrent.futures
import torch.distributed as dist
import shutil
from .utils import VARUNA_TEMP_FOLDER

try:
    from apex import amp
    from apex.amp import _amp_state
    import amp_C, apex_C
except:
    pass
opt_state_format = "opt-state-{}"
params_format = "opt-fp32-params-{}"
MARKERS = "markers"
opt_extra_state_name = "opt-common-state"

""" Writes a varuna checkpoint with model parameters, optimizer state etc. 
    Each checkpoint is a directory, written under the given path.
    
    Args:
    global_store: string, path to a folder accessible by all nodes/ranks in the training job. 
            For example, path to a mounted blob storage. This is where the varuna checkpoint folder is written.
    step: int, iteration number for checkpoint. If None, it'll be taken from varuna's tracked progress.
    tempdir: string, path to a local directory to which to write checkpoints temporarily, and sync
            with the global store in the background. Lowers checkpoint write time in the critical path.
    shard: bool, whether to shard checkpoint writes over data parallel workers as well. Speeds up checkpoint 
"""
def write_varuna_checkpoint(varuna_model, global_store, step, tempdir=None, shard=False):

    optimizer = varuna_model.optimizer
    cp_time = time.time()
    mv_futures = []
    executor = None if tempdir is None else concurrent.futures.ThreadPoolExecutor()
    rank = varuna_model.rank
    local_rank = varuna_model.local_rank
    stage = varuna_model.stage
    parameter_names = varuna_model.parameter_names
    param_name_to_pstage = varuna_model.param_name_to_pstage
    cuts_per_stage = varuna_model.partitioned_model.cuts_per_stage

    rank_within_stage = varuna_model.stage_to_rank_map[stage].index(rank)
    pstages = range(cuts_per_stage * stage, (stage+1)* cuts_per_stage)
    data_depth = len(varuna_model.stage_to_rank_map[stage])

    cp_dir_name, marker_dir_name = create_ckpt_dirs(global_store, tempdir, rank, local_rank, step)
        
    ordered_params = list(varuna_model.partitioned_model.module.parameters())   
    if varuna_model.fp16:
        ordered_params = amp.master_params(optimizer)
    mv_futures_, param_count = checkpoint_model_params(ordered_params, 
                                        rank_within_stage, shard, data_depth,
                                        pstages, parameter_names, param_name_to_pstage, 
                                        cp_dir_name, tempdir = tempdir, executor = executor)
    mv_futures.extend( mv_futures_ )
    mv_futures_, state_count = checkpoint_opt_state(optimizer, rank_within_stage, shard, data_depth,
                                            pstages, parameter_names, param_name_to_pstage, 
                                            cp_dir_name, tempdir = tempdir, executor = executor)
    mv_futures.extend( mv_futures_ )
    # assert param_count == state_count, \
    #     f"Checkpoint error! rank {rank} wrote {param_count} params but {state_count} opt states"

    # optimizer extra state
    extra_state = optimizer.state_dict()
    extra_state["state"] = {}
    torch.save(extra_state, os.path.join(cp_dir_name, opt_extra_state_name))
           
    cp_time = time.time() - cp_time
    print("Opt ckpt time", cp_time)

    ckpt_future = None
    if tempdir is not None and len(mv_futures) > 0:
        ckpt_future = executor.submit(future_on_futures, mv_futures, rank, local_rank, 
                        step, global_store, param_count)
        executor.shutdown(wait = False)
    else:
        local_tracker = get_local_ckpt_tracker(local_rank)
        with open(local_tracker,"w") as f:
            f.write(str(step))
        global_tracker = get_global_ckpt_tracker(global_store, rank, step)
        with open(global_tracker,"w") as f:
            f.write(str(param_count))

    return ckpt_future


def checkpoint_opt_state(optimizer, rank_within_stage, shard, data_depth,
                pstages, parameter_names, param_name_to_pstage, 
                cp_dir_name, tempdir = None, executor = None):
    data_depth = data_depth if shard else 1 
    mv_futures = []
    state_count = 0

    # write from the first replica of the stage or shard
    if rank_within_stage == 0 or shard:
        # save param states for each cutpoint separately
        pstage_state_dicts = dict()
        for i in pstages:
            pstage_state_dicts[i] = dict()

        # each worker has the same ordered set of state keys
        for ind, key in enumerate(optimizer.state):
            # shard over stage replicas
            if ind % data_depth != rank_within_stage:
                continue
            # store state by param names
            param_name = parameter_names[key]
            assert param_name in param_name_to_pstage, \
                   "param {} not found in rank {}".format(param_name, dist.get_rank())
            pstage = param_name_to_pstage[param_name]
            param_state = optimizer.state[key]
            pstage_state_dicts[pstage][param_name] = param_state
            state_count += 1

        for i in pstages:
            cp_name = os.path.join(cp_dir_name, opt_state_format.format(i))
            if data_depth > 1:
                cp_name += "_" + str(rank_within_stage)
            if tempdir is not None:
                temp_name =  os.path.join(tempdir, opt_state_format.format(i))
                if data_depth > 1:
                    temp_name += "_" + str(rank_within_stage)
                torch.save(pstage_state_dicts[i], temp_name)
                mv_futures.append(executor.submit(shutil.move, temp_name, cp_name))
            else:
                torch.save(pstage_state_dicts[i], cp_name)
            

    return mv_futures, state_count


def checkpoint_model_params(ordered_params, rank_within_stage, shard, data_depth,
                pstages, parameter_names, param_name_to_pstage, 
                cp_dir_name, tempdir = None, executor = None):
    data_depth = data_depth if shard else 1 
    mv_futures = []
    param_count = 0

    # write from the first replica of the stage or shard
    if rank_within_stage == 0 or shard:    
        pstage_state_dicts = dict()
        for i in pstages:
            pstage_state_dicts[i] = dict()

        for ind, p in enumerate(ordered_params):
            if ind % data_depth != rank_within_stage:
                continue
            param_name = parameter_names[p]
            # not a part of the worker's stage
            if param_name not in param_name_to_pstage:
                continue
            pstage = param_name_to_pstage[param_name]
            if pstage not in pstages:
                continue
            pstage_state_dicts[pstage][param_name] = p
            param_count += 1
        
        
        for i in pstages:
            cp_name = os.path.join(cp_dir_name, params_format.format(i))
            if data_depth > 1:
                cp_name += "_" + str(rank_within_stage)
            if tempdir is not None:
                temp_name =  os.path.join(tempdir, params_format.format(i))
                if data_depth > 1:
                    temp_name += "_" + str(rank_within_stage)
                torch.save(pstage_state_dicts[i], temp_name)
                mv_futures.append(executor.submit(shutil.move, temp_name, cp_name))
            else:
                torch.save(pstage_state_dicts[i], cp_name)
    
    return mv_futures, param_count


def create_ckpt_dirs(global_store, tempdir, rank, local_rank, step):
    cp_dir_name = os.path.join(global_store, "varuna_ckpt_{}".format(step))
    marker_dir_name = os.path.join(cp_dir_name, MARKERS)
    if rank == 0 and (not os.path.exists(cp_dir_name)):
        os.makedirs(cp_dir_name)
        os.makedirs(marker_dir_name)
    if local_rank == 0 and (tempdir is not None) and (not os.path.exists(tempdir)):
        os.makedirs(tempdir)
    while not os.path.exists(marker_dir_name):
        pass
    return cp_dir_name, marker_dir_name

def future_on_futures(mv_futures, rank, local_rank, iteration, global_store, param_count):
    done, notdone = concurrent.futures.wait(mv_futures)
    print("{} futures done!".format(len(done)))
    error = False
    if len(notdone) > 0:
        print("{} ckpts not moved\n".format(notdone))
        error = True
    for future in done:
        try:
            data = future.result()
        except Exception as exc:
            print('future generated an exception: %s' % ( exc))
            error = True
    if not error:
        local_tracker = get_local_ckpt_tracker(local_rank)
        with open(local_tracker,"w") as f:
            f.write(str(iteration))
        global_tracker = get_global_ckpt_tracker(global_store, rank, iteration)
        with open(global_tracker,"w") as f:
            f.write(str(param_count))

def load_varuna_checkpoint(my_stage, num_stages, total_num_pstages, common_store, 
                            pstages_to_read = None, device = 'cpu'):
    state_dict = {}
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    for i in pstages_to_read:
        cp_file = os.path.join(common_store, params_format.format(i))
        if os.path.exists(cp_file):
            state_dict_ = torch.load(cp_file,map_location=device)
            state_dict.update(state_dict_)
        else:
            shards = [os.path.join(common_store,f) for f in os.listdir(common_store) \
                        if f.startswith(params_format.format(i) + "_")]
            for cp_file in shards:
                state_dict_ = torch.load(cp_file,map_location=device)
                state_dict.update(state_dict_)
    return state_dict



def load_varuna_optimizer(optimizer, my_stage, num_stages, total_num_pstages, parameter_names, \
                        common_store, pstages_to_read = None, device='cpu'):
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    # reload state
    opt_state = {}
    for i in pstages_to_read:
        f = os.path.join(common_store, opt_state_format.format(i))
        if os.path.exists(f):
            state_ = torch.load(f,map_location=device)
            opt_state.update(state_)
        else:
            shards = [os.path.join(common_store,f) for f in os.listdir(common_store) \
                        if f.startswith(opt_state_format.format(i) + "_")]
            for filename in shards:
                state_ = torch.load(filename,map_location=device)
                opt_state.update(state_)
                
    for p in amp.master_params(optimizer):
        name = parameter_names[p]
        if name in opt_state:
            optimizer.state[p] = opt_state[name]
        else:
            print(f"checkpoint didn't find state for {name}")
    
    extra_state = torch.load(os.path.join(common_store, opt_extra_state_name))
    for i,g in enumerate(extra_state['param_groups']):
        for k,v in g.items():
            if k != 'params':
                optimizer.param_groups[i][k] = v


def get_local_ckpt_tracker(local_rank):
    return os.path.join(VARUNA_TEMP_FOLDER, f"ckpt_tracker_{local_rank}.txt")

def get_global_ckpt_tracker(global_store, rank, step):
    return os.path.join(global_store, "varuna_ckpt_{}".format(step),
                            MARKERS, f"complete_{rank}.txt")

def num_params_written(global_store, step):
    marker_dir = os.path.join(global_store, f"varuna_ckpt_{step}", MARKERS)
    markers = os.listdir(marker_dir)
    complete = 0
    for m in markers:
        with open(os.path.join(marker_dir, m),"r") as f:
            complete += int(f.read())
    return complete

def get_prev_checkpoint(global_store, step):
    ckpt_steps = sorted ([int(f.split("_")[-1]) for f in os.listdir(global_store)\
                         if f.startswith("varuna_ckpt_")] )
    prev_step = -1
    for c in ckpt_steps:
        if c >= step:
            break
        prev_step = c
    return prev_step

    