import torch
import torch.distributed as dist
from torch.nn import Module

from .partitioned_model import CutPoint

import os
import time
import pickle

from collections import OrderedDict 
import collections

from apex import amp
import numpy as np
import random

class Profiling:

    def __init__(self, model, device, fp16 = False):
        self.model = model
        self.ret_val = None
        self.fp16 = fp16
        self.device = device

    def initialize(self, dummy_inputs, start=0, end=1, from_cache=True):
        start_time = time.time()
        self.dry_run(dummy_inputs, from_cache)
        print("dry run time", time.time() - start_time)
        self.trim_model(start, end)
        self.check_unused_parameters(dummy_inputs)

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
    def trim_model(self, start=0, end=1):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = index
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = start
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
            if index > start and index <= end:
                used_modules.append(name)
                is_used[name] = True
            
            # only need to set up two cutpoints at most
            if isinstance(module, CutPoint):    
                if index == end:
                    attach_meta(module, start+1)
                    self.bwd_grad_shape = self.input_shapes[name][0]
                if index == start:
                    self.fwd_inp_shape = self.input_shapes[name][0]
                    used_modules.append(name)
                    is_used[name] = True
                    attach_meta(module, start)
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
    
    def check_unused_parameters(self, dummy_inputs):
        # set eval mode and clear grads
        prev_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.grad = None

        for n in self.ordered_modules:
            m = self.ordered_modules[n]
            if isinstance(m,CutPoint):
                m.set_pruning(True)

        # device = dummy_inputs[list(dummy_inputs.keys())[0]].device
        # forward
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = lambda grads=False: \
                torch.zeros(self.fwd_inp_shape, dtype=torch.float32)    
        try:
            calc_val = self.model(**dummy_inputs)
            ret_val = self.ret_val if self.ret_val is not None else calc_val
        except Exception as e:
            if self.ret_val is None:
                raise e
            ret_val = self.ret_val
        
        # backward
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = None
        ret_val.backward(torch.ones(list(ret_val.size()), dtype=torch.float32))
        

        self.ret_val = None
        to_remove = []
        for n,p in self.model.named_parameters():
            if p.grad is None:
                to_remove.append(n)
                path = n.split(".")
                parent = self.model
                for i in range(len(path) - 1):
                    parent = getattr(parent, path[i])
                setattr(parent,path[-1], None)
        
        # reset grads and train mode
        for p in self.model.parameters():
            p.grad = None
        if prev_training:
            self.model.train()

        for m in self.ordered_modules:
            m = self.ordered_modules[m]
            if isinstance(m,CutPoint):
                m.set_pruning(False)

        self.model_pruned = True
    
    def set_ret_val(self, val):
        self.ret_val = val

    def recv(self, grads=False):
        if grads:
            return self.bwd_grad
        return self.fwd_inp

    def warmup(self, get_batch_fn, microbatch_sizes, optimizer):
        optimizer._amp_lazy_init()
        
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        count = 0
        while count < 100:
            batch_size = random.choice(microbatch_sizes)
            warmup_time = time.time()
            try:
                inputs = get_batch_fn(batch_size)
                #input_mem = torch.cuda.memory_allocated() - input_mem
                if self.fwd_inp_shape is not None:
                    fwd_inp_shape = list(self.fwd_inp_shape)
                    fwd_inp_shape[0] = batch_size
                    self.fwd_inp = torch.ones(fwd_inp_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)
                torch.set_grad_enabled(True)
                try:
                    calc_val = self.model(**inputs)
                    fwd_out = self.ret_val if self.ret_val is not None else calc_val
                    del calc_val
                except Exception as e:
                    if self.ret_val is None:
                        print("Calc error!!!")
                        raise e
                    fwd_out = self.ret_val
                
                if isinstance(fwd_out, tuple):
                    fwd_out = fwd_out[0]
                grads = 0.00001 * torch.ones(list(fwd_out.size()), device = self.device)
                if self.bwd_grad_shape is not None:
                    self.bwd_grad = torch.ones(self.bwd_grad_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)

                # print("grads before",optimizer.param_groups[0]["params"][0].grad)
                if self.fp16:
                    with amp.scale_loss(fwd_out, optimizer, last_partition=True) as scaled_out:
                        scaled_out.backward(grads)
                else:
                    fwd_out.backward(grads)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Out of memorryyyy", batch_size)
                else:
                    raise e
            
            for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
            for param in self.model.parameters():
                param.grad = None

            #fwd_act_size = fwd_out.element_size() * fwd_out.nelement()
            #del grads, fwd_out
            self.model.zero_grad()
            optimizer.zero_grad()
            self.ret_val = None
            self.fwd_inp = None; self.bwd_grad = None

            warmup_time = time.time() - warmup_time
            print("warming up", warmup_time)
            count += 1

    def profile(self, get_batch_fn,  microbatch_sizes, optimizer, filename="profile.csv"):

        self.warmup(get_batch_fn, microbatch_sizes, optimizer)
        profile = {}
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        # should be 0?
        initial_mem = torch.cuda.memory_allocated(self.device)
        print("initial mem", initial_mem)

        #self.model.to(self.device)
        # model_mem = torch.cuda.memory_allocated(self.device) - initial_mem
        # print("Model memory", model_mem)

        of = open(filename,"a")
        of.write("Batch size, fwd_time, bwd_time, max_mem_usage, input_mem, fwd_act_size, opt_state_mem\n")

        for batch_size in microbatch_sizes:
            self.model.train()
            fwd_times = []
            bwd_times = []
            avg_mem_usage = 0
            try:
                for i in range(5): 
                    torch.cuda.reset_max_memory_allocated(self.device)
                    pre_mem = torch.cuda.memory_allocated()
                    # get_batch_fn should load inputs into device and return dict
                    input_mem = torch.cuda.memory_allocated()
                    inputs = get_batch_fn(batch_size)
                    input_mem = torch.cuda.memory_allocated() - input_mem
                    if self.fwd_inp_shape is not None:
                        fwd_inp_shape = list(self.fwd_inp_shape)
                        fwd_inp_shape[0] = batch_size
                        self.fwd_inp = torch.ones(fwd_inp_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)

                    start = time.time()
                
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
                    _tosend = fwd_out.cpu()
                    torch.cuda.synchronize()
                    fwd_time = time.time() - start

                    bwd_time = time.time()
                    if isinstance(fwd_out, tuple):
                        fwd_out = fwd_out[0]
                    grads = 0.00001 * torch.ones(list(fwd_out.size()), device = self.device)
                    if self.bwd_grad_shape is not None:
                        self.bwd_grad = torch.ones(self.bwd_grad_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)
                
                    # print("grads before",optimizer.param_groups[0]["params"][0].grad)
                    if self.fp16:
                        with amp.scale_loss(fwd_out, optimizer, last_partition=True) as scaled_out:
                            scaled_out.backward(grads)
                    else:
                        fwd_out.backward(grads)
                    _tosend = grads.cpu()
                    torch.cuda.synchronize()
                    bwd_time = time.time() - bwd_time

                    optimizer.step()
                    for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                        param.grad = None
                    for param in self.model.parameters():
                        param.grad = None
                    
                    fwd_act_size = fwd_out.element_size() * fwd_out.nelement()
                    del grads, fwd_out
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    mem_usage = torch.cuda.max_memory_allocated(self.device)
                
                    fwd_times.append(fwd_time)
                    bwd_times.append(bwd_time)
                    avg_mem_usage += mem_usage

                    del inputs
                    self.fwd_inp = None; self.bwd_grad = None
                
                    opt_state_mem = torch.cuda.memory_allocated(self.device)
                    optimizer.state = collections.defaultdict(dict) # Reset state
                    opt_state_mem = opt_state_mem - torch.cuda.memory_allocated(self.device)
                
                print("fwd:", fwd_times, 'bwd:', bwd_times)
                fwd_times = remove_outliers(fwd_times)
                bwd_times = remove_outliers(bwd_times)
                fwd_time = sum(fwd_times)/len(fwd_times)
                bwd_time = sum(bwd_times)/len(bwd_times)
                mem_usage = avg_mem_usage / 5

                # print("Batch size", batch_size, ": ")
                # print("fwd_time", fwd_time)
                # print("bwd_time", bwd_time)
                # print("mem_usage", mem_usage)
                # print("-------------------------")
                # print()
                
                string = "{}, {}, {}, {}, {}, {}, {}".format(\
                                    batch_size, fwd_time, bwd_time, 
                                    mem_usage, input_mem, fwd_act_size, 
                                    opt_state_mem)
                print(string)
                of.write(string + "\n")
                # print("Opt_state", opt_state_mem)

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


def remove_outliers(data):
    #define a list to accumlate anomalies
    anomalies = []
    
    n = len(data)
    data = sorted(data)
    if n % 2:
        median = data[n//2]
    else:
        median = (data[n//2] + data[(n//2)-1])/2
     
    upper_limit = 3*median
    # print(lower_limit)
    filtered = []
    for point in data:
        if point <= upper_limit:
            filtered.append(point)
    return filtered
