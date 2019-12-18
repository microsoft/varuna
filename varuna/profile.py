import torch
import torch.distributed as dist
from torch.nn import Module

from .partitioned_model import CutPoint

import os
import time
import pickle

from collections import OrderedDict 

blob_store_folder = "~/myblobcontainer"

cp_partition_prefix = "cp-partition"

class Profiling:

    def __init__(self, model, device):
        self.model = model
        self.ret_val = None
        self.device = device

    def initialize(self, dummy_inputs, num_stages=1, from_cache=False):
        start = time.time()
        self.dry_run(dummy_inputs)
        print("dry run time", time.time() - start)
        self.trim_model(k=num_stages)

    def dry_run(self, dummy_inputs):
        # executes the forward pass of the module on dummy inputs. 
        # Sets the order in which modules are used and the total number of cutpoints declared.

        self.ordered_modules = OrderedDict()
        self.num_cutpoints = 0

        # if not (from_cache and os.path.exists("_tmp_ord_mod")):

        def get_hook(name):
            def add_module_hook(module, inputs, _output):
                self.ordered_modules[name] = module
            return add_module_hook

        modules = self.model.named_modules()
        hooks = []

        for name, module in modules:
            if name == "":
                continue
            hooks.append( module.register_forward_hook(get_hook(name)))
            if isinstance(module, CutPoint):
                self.num_cutpoints += 1
        
        self.model(**dummy_inputs)

        for h in hooks:
            h.remove()
    
# """ Trims out the first stage from model. """
    def trim_model(self, k=1):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = 1
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = 0
            cutpoint.device = self.device
            cutpoint.send_fn = lambda x: None
            cutpoint.set_cp_func()

        modules = self.ordered_modules
        index = 1

        used_modules = []
        is_used = {}

        for name in modules:
            if name == "":
                continue
            module = modules[name]
            is_used[name] = False
            if index <= k:
                if isinstance(module, CutPoint):    
                    if index == k:
                        attach_meta(module, index)
                    index += 1
                used_modules.append(name)
                is_used[name] = True

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
                modules = self.model._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                modules[path[-1]] = None
                modules[path[-1]] = PassThroughModule()
                self.ordered_modules[path[-1]] = None

    def set_ret_val(self, val):
        self.ret_val = val
    

    def profile(self, get_batch_fn,  microbatch_sizes, optimizer, filename="profile.csv"):

        profile = {}

        initial_mem = torch.cuda.memory_allocated(self.device)
        self.model.to(self.device)
        model_mem = torch.cuda.memory_allocated(self.device) - initial_mem
        print("Model memory", model_mem)

        of = open(filename,"w")
        of.write("Batch size, fwd_time, bwd_time, max_mem_usage, input_mem, model_mem, post_fwd_bwd_mem\n")

        for batch_size in microbatch_sizes:
            self.model.train()
            try: 
                torch.cuda.reset_max_memory_allocated(self.device)
                # get_batch_fn should load inputs into device and return dict
                input_mem = torch.cuda.memory_allocated()
                inputs = get_batch_fn(batch_size)
                print(inputs["input_ids"].size())
                input_mem = torch.cuda.memory_allocated() - input_mem
                start = time.time()

                post_fwd_bwd_mem = torch.cuda.memory_allocated()
                try:
                    calc_val = self.model(**inputs)
                    fwd_out = self.ret_val if self.ret_val is not None else calc_val
                    del calc_val
                except Exception as e:
                    if self.ret_val is None:
                        print("Calc error!!!")
                        raise e
                    fwd_out = self.ret_val
                self.ret_val = None
                fwd_time = time.time() - start

                bwd_time = time.time()
                grads = torch.ones(list(fwd_out.size()), device = self.device)
                fwd_out.backward(grads)
                bwd_time = time.time() - bwd_time

                update_time = time.time()
                optimizer.step()
                del grads, fwd_out
                post_fwd_bwd_mem = torch.cuda.memory_allocated() - post_fwd_bwd_mem
                self.model.zero_grad()
                optimizer.zero_grad()
                update_time = time.time() - update_time

                mem_usage = torch.cuda.max_memory_allocated(self.device)
                profile[batch_size] = {
                    "fwd_time": fwd_time,
                    "bwd_time": bwd_time,
                    "max_mem_usage": mem_usage,
                    "input_mem": input_mem,
                    "model_mem": model_mem,
                    "post_fwd_bwd_mem": post_fwd_bwd_mem
                }
                of.write("{}, {}, {}, {}, {}, {}, {}\n".format(batch_size, fwd_time, bwd_time, mem_usage, input_mem, model_mem, post_fwd_bwd_mem))
                print("Batch size", batch_size, ": ")
                print("fwd_time", fwd_time)
                print("bwd_time", bwd_time)
                print("mem_usage", mem_usage)
                print("-------------------------")
                print()
                del inputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Out of memorryyyy")
                    break
                else:
                    raise e

        of.close()


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None
