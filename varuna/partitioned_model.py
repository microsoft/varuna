import torch
import torch.distributed as dist
from torch.nn import Module

import os
import time
import pickle

from collections import OrderedDict 

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
        self.pruning = False
    
    def set_pruning(self, boolean):
        self.pruning = boolean

    def forward(self, *inputs, **kwargs):
        # not set by ModelParallel, pass through as is
        if self.cp_func is None:
            return inputs[0]

        if len(inputs) < 0 or (len(inputs) == 1 and inputs[0] is None):
            if self.pruning:
                inputs = (torch.tensor([-1.0],requires_grad = True),)
            else:
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
                    # recv_time = time.time()
                    i = self.recv_fn()
                    # recv_time = time.time() - recv_time
                # send activations
                elif is_in_prev_stage and self.send_fn is not None:
                    self.send_fn(i)
                return i

            @staticmethod
            def backward(ctx, grad_output):
                # receive gradients.
                if is_in_prev_stage and self.recv_fn is not None:
                    # recv_time = time.time()
                    grad_output = self.recv_fn(grads = True)
                    # recv_time = time.time() - recv_time
                    # self.logfile.write("rcv grads " + str(recv_time) + "\n")
                # send gradients
                elif is_in_next_stage and self.send_fn is not None:
                    self.send_fn(grad_output, grads = True)
                return grad_output
        
        c = CutpointFunction()
        self.cp_func = c


class PartitionedModel(Module):

    def __init__(self, module, rank, local_rank, device, stage_to_rank_map, fp16, shared_weights=None):
        super(PartitionedModel, self).__init__()
        self.module = module
        self.num_stages = len(stage_to_rank_map)
        self.stage_to_rank_map = stage_to_rank_map
        self.rank = rank
        self.local_rank = local_rank
        self.fp16 = fp16
        self.shared_weights = shared_weights
        
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

        # self.logfile = open("wait_logs" + str(self.rank),"w")

    def initialize(self, dummy_inputs, from_cache=False):
        # print("Initializing partitioned model!")
        start = time.time()
        self.dry_run(dummy_inputs, from_cache)
        if self.shared_weights is not None:
            self.find_shared_weight_stages()
        print("dry run time", time.time() - start)
        self.prep_cutpoints()
        self.remove_unused_parameters(dummy_inputs)
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
            
            with torch.no_grad():
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
    
    def find_shared_weight_stages(self):

        all_shared_weights = []
        for w_pair in self.shared_weights:
            all_shared_weights += [w for w in w_pair]
        curr_stage = 0
        weight_stages = dict()
        for m in self.ordered_modules:
            module = self.ordered_modules[m]
            if isinstance(module, CutPoint):
                curr_stage += 1
                continue
            for w in all_shared_weights:
                param_name = w.split(".")[-1]
                module_name = w[ : -len(param_name)-1]
                if m == module_name and hasattr(module, param_name):
                    weight_stages[w] = curr_stage
                    break
                elif m == module_name:
                    print("Here we have the peculiar case of the missing weight", m, param_name)
                    print(getattr(module,param_name))
        
        for w in all_shared_weights:
            if w not in weight_stages:
                param_name = w.split(".")[-1]
                if hasattr(self.module, param_name):
                    weight_stages[w] = curr_stage

        cuts_per_stage = (self.num_cutpoints + 1)/ self.num_stages
        shared_weight_stages = []
        for w_pair in self.shared_weights:
            for w in w_pair:
                assert w in weight_stages, "Shared parameter {} not found in model!".format(w)
                weight_stages[w] = int(weight_stages[w] // cuts_per_stage)
            shared_weight_stages.append((weight_stages[w_pair[0]], weight_stages[w_pair[1]]))

        self.shared_weight_stages = shared_weight_stages


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
    def remove_unused_parameters(self, dummy_inputs):

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

        self.check_unused_parameters(dummy_inputs)

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
        param_name_to_pstage = dict()
        temp_param_names = []

        for name in modules:
            module = self.ordered_modules[name]
            if name == "" or module is None:
                continue
            if isinstance(module, CutPoint):
                if len(state_dict.keys()) > 0:
                    # torch.save(state_dict, os.path.join(checkpoint_dir, "cp-pstage-{}".format(str(stage_index))))
                    for p in temp_param_names:
                        param_name_to_pstage[p] = stage_index
                    temp_param_names = []
                    cp_count += 1
                if cp_count >= self.cuts_per_stage:
                    break
                stage_index += 1
                state_dict = {}
            else:
                module_state_dict = module.state_dict()
                module_state_dict_ = OrderedDict()
                for key, val in module_state_dict.items():
                    param_name = name + "." + key
                    module_state_dict_[param_name] = val
                    temp_param_names.append(param_name)

                state_dict.update(module_state_dict_)

        # last cutpoint
        if len(state_dict.keys()) > 0 and (cp_count < self.cuts_per_stage):
            state_dict["lm_head_weight"] = self.module.lm_head_weight
            # torch.save(state_dict, os.path.join(checkpoint_dir, "cp-pstage-{}".format(str(stage_index))))
            for p in temp_param_names:
                param_name_to_pstage[p] = stage_index
            # param_name_to_pstage["lm_head_weight"] = stage_index
            
        print("checkpointed!!")
        return param_name_to_pstage

    def check_unused_parameters(self, dummy_inputs):
        # set eval mode and clear grads
        prev_training = self.module.training
        self.module.eval()
        for p in self.module.parameters():
            p.grad = None

        for n in self.ordered_modules:
            m = self.ordered_modules[n]
            if isinstance(m,CutPoint):
                m.set_pruning(True)

        # forward
        self.set_recv_fn(lambda grads=False: torch.zeros(self.forward_input_shapes[0], dtype=torch.float32))     
        try:
            calc_val = self.module(**dummy_inputs)
            ret_val = self.ret_val if self.ret_val is not None else calc_val
        except Exception as e:
            if self.ret_val is None:
                raise e
            ret_val = self.ret_val
        
        # backward
        self.set_recv_fn(None)
        if self.stage != self.num_stages - 1:
            ret_val.backward(torch.ones(list(ret_val.size()), dtype=torch.float32))
        else:
            ret_val.backward()

        self.ret_val = None
        to_remove = []
        for n,p in self.module.named_parameters():
            if p.grad is None:
                to_remove.append(n)
                path = n.split(".")
                parent = self.module
                for i in range(len(path) - 1):
                    parent = getattr(parent, path[i])
                setattr(parent,path[-1], None)
        
        # reset grads and train mode
        for p in self.module.parameters():
            p.grad = None
        if prev_training:
            self.module.train()

        for m in self.ordered_modules:
            m = self.ordered_modules[m]
            if isinstance(m,CutPoint):
                m.set_pruning(False)

        self.model_pruned = True

        
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
        # self.logfile.flush()
        self.ret_val = None
        return ret_val 


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None


def load_varuna_checkpoint(my_stage, num_stages, total_num_pstages, common_store,prefix="cp-pstage"):
    state_dict = {}
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
