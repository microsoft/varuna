import torch
import torch.distributed as dist
from torch.nn import Module

import os
import time
import pickle

from collections import OrderedDict 

blob_store_folder = "~/myblobcontainer"

cp_partition_prefix = "cp-partition"


class CutPoint(Module):
    def __init__(self):
        super(CutPoint, self).__init__()
        # start with 1 and end before last stage (total num_stages - 1 )
        self.cp_index = -1
        self.cp_func = None
        
        self.set_ret_val_func = None
        self.device = None
        self.send_fn = self.recv_fn = None
        self.stage = -1
        self.fp16 = False


    def forward(self, *inputs, **kwargs):
        # not set by ModelParallel, pass through as is
        if self.cp_func is None:
            return inputs[0]

        if len(inputs) < 0 or (len(inputs) == 1 and inputs[0] is None):
            dtype = torch.float16 if self.fp16 else torch.float32
            inputs = (torch.tensor([-1.0],requires_grad = True, dtype=dtype).to(self.device),)

        if isinstance(self.cp_func, torch.autograd.Function):
            out = self.cp_func.apply(*inputs, **kwargs)            
            if self.cp_index == (self.stage + 1):
                self.set_ret_val_func(out) 
            return out
        
        return self.cp_func(*inputs, **kwargs)

    def set_cp_func(self):
        
        is_in_next_stage = self.cp_index == self.stage
        is_in_prev_stage = self.cp_index == (self.stage + 1)

        class CutpointFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                # recieve activations
                if is_in_next_stage and self.recv_fn is not None:
                    i = self.recv_fn()
                # send activations
                elif is_in_prev_stage:
                    self.send_fn(i)
                return i

            @staticmethod
            def backward(ctx, grad_output):
                # receive gradients.
                if is_in_prev_stage and self.recv_fn is not None:
                    grad_output = self.recv_fn(grads = True)
                # send gradients
                elif is_in_next_stage:
                    self.send_fn(grad_output, grads = True)
                return grad_output
        
        c = CutpointFunction()
        self.cp_func = c


class PartitionedModel(Module):

    def __init__(self, module, rank, local_rank, device, stage_to_rank_map, fp16):
        super(PartitionedModel, self).__init__()
        self.module = module
        self.is_data_parallel = False
        self.num_stages = len(stage_to_rank_map)
        self.stage_to_rank_map = stage_to_rank_map
        self.rank = rank
        self.local_rank = local_rank
        self.fp16 = fp16
        
        torch.cuda.set_device(device)
        self.device = torch.device("cuda", device)

        self.ret_val = None
        self.pre_cp = None
        self.post_cp = None

        self.stage = -1
        for stage in self.stage_to_rank_map:
            if self.rank in self.stage_to_rank_map[stage]:
                self.stage = stage
                break
        else:
            raise ValueError("Rank " + self.rank + " not found in stage to rank map!")

    def initialize(self, dummy_inputs, from_cache=False):
        # print("Initializing partitioned model!")
        start = time.time()
        self.dry_run(dummy_inputs, from_cache)
        print("dry run time", time.time() - start)
        self.prep_cutpoints()
        self.remove_unused_parameters()
        self.model_pruned = True


    def dry_run(self, dummy_inputs, from_cache):
        # """ executes the forward pass of the module on dummy inputs. Sets the order in which modules are used and the total number of cutpoints declared. """
        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        if self.local_rank == 0 and not (from_cache and os.path.exists("_tmp_ord_mod") and os.path.exists("_tmp_inp_shapes")):
            # store input shapes for each module (or atleast each cp)
            print("Initializing partitioned model!")

            def get_hook(name):
                def add_module_hook(module, inputs, _output):
                    self.ordered_modules[name] = module
                    if isinstance(module, CutPoint):
                        # if len(inputs) > 1: error
                        self.input_shapes[name] = [list(inputs[0].size())]
                return add_module_hook

            modules = self.module.named_modules()
            hooks = []

            for name, module in modules:
                if name == "":
                    continue
                hooks.append( module.register_forward_hook(get_hook(name)))
                if isinstance(module, CutPoint):
                    self.num_cutpoints += 1
            
            self.module(**dummy_inputs)

            for h in hooks:
                h.remove()

            # actually just need the order as a list of names
            with open("_tmp_ord_mod",'wb') as f:
                pickle.dump(list(self.ordered_modules.keys()),f)
            with open("_tmp_inp_shapes",'wb') as f:
                pickle.dump(self.input_shapes,f)
            dist.barrier()

        else:
            dist.barrier() 
            with open("_tmp_ord_mod",'rb') as f:
                ordered_modules = pickle.load(f)

            for n in ordered_modules:
                path = n.split(".")
                modules = self.module._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                self.ordered_modules[n] = modules[path[-1]]

            with open("_tmp_inp_shapes",'rb') as f:
                self.input_shapes = pickle.load(f)
            self.num_cutpoints = len(self.input_shapes)
    
# """ setting actual cutpoint functions for comunication. """
    def prep_cutpoints(self):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = index
            cutpoint.num_stages = self.num_stages
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = self.stage
            cutpoint.device = self.device
            cutpoint.fp16 = self.fp16
            cutpoint.set_cp_func()

        self.cuts_per_stage = (self.num_cutpoints + 1) // self.num_stages

        modules = self.ordered_modules
        index = 1
        assigned_index = 1

        self.forward_input_shapes = []
        self.backward_grad_shapes = []

        for name in modules:
            module = modules[name]
            if name == "":
                continue
            if isinstance(module, CutPoint):
                if (index % self.cuts_per_stage == 0):
                    # pre cp
                    if assigned_index == self.stage:
                        self.forward_input_shapes = self.input_shapes[name]
                        self.pre_cp = module
                    # post cp
                    if assigned_index == self.stage + 1:
                        self.backward_grad_shapes = self.input_shapes[name]
                        self.post_cp = module
                    attach_meta(module, assigned_index)
                    assigned_index += 1  
                index += 1
            # found all relevant cutpoints, break
            if assigned_index == self.num_stages:
                break
        
# """ remove unused modules to save memory. """
    def remove_unused_parameters(self):

        pre_cp_index = self.stage
        post_cp_index = self.stage + 1

        is_used = {}
        used_modules = []
        add_flag = (self.stage == 0)
        
        modules = self.ordered_modules

        for name in modules:
            module = modules[name]
            if name == "":
                continue
            if isinstance(module, CutPoint):
                if (module.cp_index == pre_cp_index or module.cp_index == post_cp_index): 
                    add_flag = not add_flag
            else:
                if add_flag:
                    used_modules.append(name)
                is_used[name] = add_flag

        # any module that is used or has children that are used are needed
        for u in used_modules:
            path = u.split(".")
            key = path[0]
            for i in range(1,len(path)):
                is_used[key] = True
                key = key + "." + path[i]

        for m in is_used:
            if not is_used[m]:
                path = m.split(".")
                modules = self.module._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                modules[path[-1]] = None
                modules[path[-1]] = PassThroughModule()
                self.ordered_modules[m] = None

        self.model_pruned = True

    """For each partition/stage, the first rank in that stage saves state_dict 
    of the cutpoints used by that stage to a common store. So checkpoint is sharded 
    along cutpoints (as many checkpoint files as cutpoints)"""
    def checkpoint(self, checkpoint_dir = "model-checkpoint"):
        # only 1 rank per stage for data ||
        if self.rank != self.stage_to_rank_map[self.stage][0]:
            return
        
        # we only want to checkpoint leaf modules (For ex. bert.embedding.weight and not bert.embeddings)
        leaf_modules = {}
        non_leaf_modules = {}

        # this assumes that a non-leaf module is added after all it's descendants to the order (which is true so far)
        for m in self.ordered_modules:
            if (m not in leaf_modules) and (m not in non_leaf_modules):
                leaf_modules[m] = True
                # mark all ancestors as non-leaf
                path = m.split(".")
                key = path[0]
                for i in range(1,len(path)):
                    non_leaf_modules[key] = True
                    key = key + "." + path[i]

        modules = list(leaf_modules.keys())

        stage_index = 0
        state_dict = {}
        cp_count = 0

        for name in modules:
            module = self.ordered_modules[name]
            if name == "" or module is None:
                continue
            if isinstance(module, CutPoint):
                if len(state_dict.keys()) > 0:
                    torch.save(state_dict, os.path.join(checkpoint_dir, "cp-pstage-{}".format(str(stage_index))))
                    cp_count += 1
                if cp_count >= self.cuts_per_stage:
                    break
                stage_index += 1
                state_dict = {}
            else:
                module_state_dict = module.state_dict()
                module_state_dict_ = OrderedDict()
                for key, val in module_state_dict.items():
                    module_state_dict_[name + "." + key] = val
                state_dict.update(module_state_dict_)

        # last cutpoint
        if len(state_dict.keys()) > 0 and (cp_count < self.cuts_per_stage):
            torch.save(state_dict, os.path.join(checkpoint_dir, "cp-pstage-{}".format(str(stage_index))))

        print("checkpointed!!")

    def set_ret_val(self, val):
        self.ret_val = val

    def set_send_fn(self, send_fn):
        if self.pre_cp is not None:
            self.pre_cp.send_fn = send_fn
        if self.post_cp is not None:
            self.post_cp.send_fn = send_fn

    def set_recv_fn(self, recv_fn):
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = recv_fn
        if self.post_cp is not None:
            self.post_cp.recv_fn = recv_fn

    def forward(self, *inputs, **kwargs):
        # if not self.model_pruned:
            # raise Error
    
        try:
            calc_val = self.module(*inputs, **kwargs)
            ret_val = self.ret_val if self.ret_val is not None else calc_val
        except Exception as e:
            if self.ret_val is None:
                raise e
            ret_val = self.ret_val

        self.ret_val = None
        return ret_val 


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None


def load_varuna_checkpoint(my_stage, num_stages, total_num_pstages, common_store):
    state_dict = {}
    stages_per_worker = total_num_pstages // num_stages
    pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    print(dist.get_rank(),"rank with stage", my_stage, "reads", pstages_to_read)
    for i in pstages_to_read:
        state_dict_ = torch.load(os.path.join(common_store, "cp-pstage-{}".format(i)),map_location="cpu")
        state_dict.update(state_dict_)
    return state_dict