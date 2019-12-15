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

    def __init__(self, module):
        self.module = module
        self.ret_val = None
        self.pre_cp = None
        self.post_cp = None
        self.device = torch.device("cuda", torch.cuda.current_device())

    def initialize(self, dummy_inputs):
        start = time.time()
        self.dry_run(dummy_inputs)
        # print("dry run time", time.time() - start)
        self.trim_model()

    def dry_run(self, dummy_inputs):
        # executes the forward pass of the module on dummy inputs. 
        # Sets the order in which modules are used and the total number of cutpoints declared.

        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        # store input shapes for each module (or atleast each cp)
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
    
# """ Trims out the forst stage from model. """
    def trim_model(self):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = index
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = 0
            cutpoint.device = self.device
            cutpoint.send_fn = lambda x: None
            cutpoint.set_cp_func()

        modules = self.ordered_modules
        index = 1

        self.forward_input_shapes = []
        self.backward_grad_shapes = []

        used_modules = []
        is_used = {}

        for name in modules:
            module = modules[name]
            is_used[name] = False
            if name == "":
                continue
            if index == 1:
                if isinstance(module, CutPoint):    
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
                modules = self.module._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                modules[path[-1]] = None
                modules[path[-1]] = PassThroughModule()
                self.ordered_modules[path[-1]] = None

    def set_ret_val(self, val):
        self.ret_val = val
    

    def profile(self, train_dataset, max_batch_size = None):
        if max_batch_size is None:
            max_batch_size = len(train_dataset)

        batch_size = 2
        profile = {}

        model.to
        mem usage so far
        
        while not OOM and batch_size < max_batch_size:
            inputs = train_dataset[:batch_size]
            inputs.to
            start = time.time()
            fwd_out = self.module(inputs)
            fwd_time = time.time() - start
            bwd_time = time.time()
            fwd_out.backward(torch.ones(list(fwd_out.size())))
            bwd_time = time.time() - bwd_time
            profile[batch_size] = {
                "fwd_time"
                "bwd_time"
                "mem_usage"
            }
            batch_size *= 2
            remove inputs from mem


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None
