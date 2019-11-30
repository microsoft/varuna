import torch
import torch.distributed as dist
from torch.nn import Module

import os
import time
import pickle

from collections import OrderedDict 

blob_store_folder = "~/myblobcontainer"


class CutPoint(Module):
    def __init__(self):
        super(CutPoint, self).__init__()
        # start with 1 and end before last stage (total num_stages - 1 )
        self.cp_index = -1
        self.num_stages = -1

        self.cp_func = None
        
        self.set_ret_val_func = None

        self.send_fn = self.recv_fn = None


    def forward(self, *inputs, **kwargs):
        # not set by ModelParallel, pass through as is
        if self.cp_func is None:
            return inputs[0]

        if len(inputs) < 0 or (len(inputs) == 1 and inputs[0] is None):
            inputs = (torch.tensor([-1.0],requires_grad = True).cuda(),)

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
                if is_in_next_stage:
                    # receive inputs.
                    inputs = self.recv_fn()
                    return inputs
                elif is_in_prev_stage:
                    # send activations
                    self.send_fn(i)
                return i

            @staticmethod
            def backward(ctx, grad_output):
                if is_in_prev_stage:
                    # receive gradients.
                    gradients = self.recv_fn(grads = True)
                    return gradients
                elif is_in_next_stage:
                    # send gradients
                    self.send_fn(grad_output, grads = True)
                return grad_output
        
        c = CutpointFunction()
        self.cp_func = c


class PartitionedModel(Module):

    def __init__(self, module, rank, local_rank, device, stage_to_rank_map):
        super(PartitionedModel, self).__init__()
        # print("Got device", device,"!!")
        self.module = module
        self.num_stages = len(stage_to_rank_map)
        self.stage_to_rank_map = stage_to_rank_map
        self.rank = rank
        self.local_rank = local_rank
        # torch.cuda.set_device(device)
        self.device = device

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

    def initialize(self, dummy_inputs):
        # print("Initializing partitioned model!")
        start = time.time()
        self.dry_run(dummy_inputs)
        print("dry run time", time.time() - start)
        self.prep_cutpoints()
        self.remove_unused_parameters()
        self.model_pruned = True
    
    def dry_run(self, dummy_inputs):
        # """ executes the forward pass of the module on dummy inputs. Sets the order in which modules are used and the total number of cutpoints declared. """
        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        if self.local_rank == 0:
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
                pickle.dump(self.ordered_modules,f)
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

        # print(self.rank, "uses", len(used_modules), "out of", len(self.ordered_modules))

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
                self.ordered_modules[path[-1]] = None

        self.model_pruned = True

    def checkpoint(self, checkpoint_name = "model-checkpoint"):
        if self.rank != 0:
            file_name = checkpoint_name + "-" + str(self.rank)
            torch.save(self.module.state_dict(), file_name)
            os.system("sudo mv " +  file_name + " " + os.path.join(blob_store_folder, file_name))
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            complete_state_dict = self.module.state_dict()

            # gather and read all local pickles to form combined state_dicts
            for i in range(self.num_stages):
                for rank in self.stage_to_rank_map[i]:
                    if rank == 0:
                        continue
                    file_name = checkpoint_name + "-" + str(rank)
                    os.system("sudo mv " + os.path.join(blob_store_folder, file_name) + " " + file_name)
                    state_dict = torch.load(file_name)
                    for key in state_dict:
                        if key not in complete_state_dict or complete_state_dict[key] == None:
                            complete_state_dict[key] = state_dict[key]
                    os.system("sudo rm " + file_name)

            # for key in complete_state_dict:
            #     print(key)

            torch.save(complete_state_dict, checkpoint_name)
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
            ret_val = self.ret_val
        except Exception as e:
            if self.ret_val is None:
                raise e
            ret_val = self.ret_val

        self.ret_val = None
        if ret_val is None:
            return calc_val
        return ret_val 


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None
