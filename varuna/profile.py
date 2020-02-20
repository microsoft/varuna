import torch
import torch.distributed as dist
from torch.nn import Module

from .partitioned_model import CutPoint

import os
import time
import pickle

from collections import OrderedDict 
import collections

class Profiling:

    def __init__(self, model, device):
        self.model = model
        self.ret_val = None
        self.device = device

    def initialize(self, dummy_inputs, stage_num=1, from_cache=False):
        start = time.time()
        self.dry_run(dummy_inputs, from_cache)
        print("dry run time", time.time() - start)
        self.trim_model(k=stage_num)

    def dry_run(self, dummy_inputs, from_cache):
        # executes the forward pass of the module on dummy inputs. 
        # Sets the order in which modules are used and the total number of cutpoints declared.

        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        if not (from_cache and os.path.exists("_tmp_ord_mod") and os.path.exists("_tmp_inp_shapes")):

            def get_hook(name):
                def add_module_hook(module, inputs, _output):
                    self.ordered_modules[name] = module
                    if isinstance(module, CutPoint):
                        # if len(inputs) > 1: error
                        self.input_shapes[name] = [list(inputs[0].size())]
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

            with open("_tmp_ord_mod",'wb') as f:
                pickle.dump(list(self.ordered_modules.keys()),f)
            with open("_tmp_inp_shapes",'wb') as f:
                pickle.dump(self.input_shapes,f)

        else:
            with open("_tmp_ord_mod",'rb') as f:
                ordered_modules = pickle.load(f)

            for n in ordered_modules:
                path = n.split(".")
                modules = self.model._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                self.ordered_modules[n] = modules[path[-1]]

            with open("_tmp_inp_shapes",'rb') as f:
                self.input_shapes = pickle.load(f)
            self.num_cutpoints = len(self.input_shapes)
    
# """ Trims out the kth stage (starting from 1) from model. """
    def trim_model(self, k=1):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = index
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = k-1
            cutpoint.device = self.device
            cutpoint.send_fn = lambda x,grads=False: None
            cutpoint.set_cp_func()

        modules = self.ordered_modules
        index = 1

        self.fwd_inp_shape = None
        self.bwd_grad_shape = None

        self.pre_cp = None

        used_modules = []
        is_used = {}

        for name in modules:
            if name == "":
                continue
            module = modules[name]

            is_used[name] = False
            # when the next cutpoint to come is the kth, modules are used
            if index == k:
                used_modules.append(name)
                is_used[name] = True
            
            # only need to set up two cutpoints at most
            if isinstance(module, CutPoint):    
                if index == k:
                    attach_meta(module, index)
                    self.bwd_grad_shape = self.input_shapes[name][0]
                if index == k-1:
                    self.fwd_inp_shape = self.input_shapes[name][0]
                    used_modules.append(name)
                    is_used[name] = True
                    attach_meta(module, index)
                    self.pre_cp = module
                index += 1
            

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
                self.ordered_modules[m] = None
    
    def set_ret_val(self, val):
        self.ret_val = val

    def recv(self, grads=False):
        if grads:
            return self.bwd_grad
        return self.fwd_inp

    def profile(self, get_batch_fn,  microbatch_sizes, optimizer, filename="profile.csv"):

        profile = {}
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        # should be 0?
        initial_mem = torch.cuda.memory_allocated(self.device)
        print("initial mem", initial_mem)
        self.model.to(self.device)
        model_mem = torch.cuda.memory_allocated(self.device) - initial_mem
        print("Model memory", model_mem)

        of = open(filename,"w")
        of.write("Batch size, fwd_time, bwd_time, max_mem_usage, input_mem, model_mem, pre_fwd_bwd_mem\n")

        for batch_size in microbatch_sizes:
            self.model.train()
            try: 
                torch.cuda.reset_max_memory_allocated(self.device)
                print("Pre pre mem", torch.cuda.memory_allocated())
                # get_batch_fn should load inputs into device and return dict
                input_mem = torch.cuda.memory_allocated()
                inputs = get_batch_fn(batch_size)
                input_mem = torch.cuda.memory_allocated() - input_mem
                if self.fwd_inp_shape is not None:
                    self.fwd_inp = torch.ones(self.fwd_inp_shape, dtype = torch.float32).to(self.device)

                start = time.time()
                pre_fwd_bwd_mem = torch.cuda.memory_allocated()
                
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
                if isinstance(fwd_out, tuple):
                    fwd_out = fwd_out[0]
                grads = torch.ones(list(fwd_out.size()), device = self.device)
                if self.bwd_grad_shape is not None:
                    self.bwd_grad = torch.ones(self.bwd_grad_shape, dtype = torch.float32).to(self.device)
                
                fwd_out.backward(grads)
                bwd_time = time.time() - bwd_time

                optimizer.step()
                

                del grads, fwd_out
                self.model.zero_grad()
                optimizer.zero_grad()

                mem_usage = torch.cuda.max_memory_allocated(self.device)
                profile[batch_size] = {
                    "fwd_time": fwd_time,
                    "bwd_time": bwd_time,
                    "max_mem_usage": mem_usage,
                    "input_mem": input_mem,
                    "model_mem": model_mem,
                    "pre_fwd_bwd_mem": pre_fwd_bwd_mem
                }
                of.write("{}, {}, {}, {}, {}, {}, {}\n".format(batch_size, fwd_time, bwd_time, mem_usage, input_mem, model_mem ,pre_fwd_bwd_mem))
                print("Batch size", batch_size, ": ")
                print("fwd_time", fwd_time)
                print("bwd_time", bwd_time)
                print("mem_usage", mem_usage)
                print("-------------------------")
                print()
                del inputs
                self.fwd_inp = None; self.bwd_grad = None
                optimizer.state = collections.defaultdict(dict) # Reset state

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
