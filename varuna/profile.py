import torch
import torch.distributed as dist
from torch.nn import Module

from .partitioned_model import CutPoint

import os
import time
import pickle
import math

from collections import OrderedDict 
import collections

try:
    from apex import amp
except:
    print("No apex")
    
import numpy as np
import random, math

from queue import Queue
from threading import Thread

num_compute_passes = 5

def remove_outliers(times, error_margin = 0.3):
    times = sorted(times)
    dp = len(times)
    mid_i = dp // 2
    median_time = times[mid_i]
    filtered_times = []
    if dp % 2 == 0:
        median_time = (median_time + times[mid_i-1]) / 2
    for t in times:
        error = abs((t-median_time)/median_time)
        if error < error_margin:
            filtered_times.append(t)
    return filtered_times

def receiver(recv_rank, recv_shape, recv_times):
    print("Start reciever from", recv_rank)
    chunks = 30
    dtype = torch.float16
    recv_handles = Queue()

    for _ in range(chunks):
        acts_tensor = torch.ones(recv_shape, dtype=dtype)
        start_time = time.time()
        handle = dist.irecv(acts_tensor, src=recv_rank)
        recv_handles.put((handle, start_time))
        if recv_handles.qsize() > 4:
            handle, start_time = recv_handles.get()
            handle.wait()
            recv_times.append(time.time() - start_time)

    while not recv_handles.empty():
        handle, start_time = recv_handles.get()
        handle.wait()
        recv_times.append(time.time() - start_time)
    
def sender(send_rank, send_shape, send_times):
    print("Start sender to",send_rank)
    chunks = 30
    dtype = torch.float16
    send_handles = Queue()

    for _ in range(chunks):
        output_acts = torch.ones(send_shape, dtype=dtype)
        start_time = time.time()
        handle = dist.isend(output_acts, dst=send_rank)
        send_handles.put((handle, start_time))
        if send_handles.qsize() > 4:
            handle, start_time = send_handles.get()
            handle.wait()
            send_times.append(time.time() - start_time)
    
    while not send_handles.empty():
        handle, start_time = send_handles.get()
        handle.wait()
        send_times.append(time.time() - start_time)

class Profiler:

    def __init__(self, model, device, fp16 = False):
        self.model = model
        self.ret_val = None
        self.fp16 = fp16
        self.device = device
        torch.cuda.set_device(device)

        dist.init_process_group(backend="gloo")

        self.rank = dist.get_rank()
        self.local_rank = int(os.getenv("LOCAL_RANK", self.rank))
        self.world_size = dist.get_world_size()

    def initialize(self, dummy_inputs, stages_to_profile=None,from_cache=True):
        self.dry_run(dummy_inputs, from_cache)
        if stages_to_profile is None:
            stages_to_profile = range(self.num_cutpoints+1)
        my_stages_to_profile = \
            list(range(self.rank, len(stages_to_profile), self.world_size))
        my_stages_to_profile = [stages_to_profile[i] for i in my_stages_to_profile]
        self.stages_to_profile = my_stages_to_profile

        self.orig_modules = dict()
        for name in self.ordered_modules:
            self.orig_modules[name] = self.ordered_modules[name]

    def profile_all(self, get_batch_fn,  microbatch_sizes, get_optimizer_fn ):

        for stage in self.stages_to_profile:
            self.stage = stage
            print("STAGE", self.stage)
            self.trim_model(self.stage, self.stage + 1)
            # self.check_unused_parameters(dummy_inputs)
            self.model.to(self.device)
            optimizer = get_optimizer_fn(self.model)

            self.profile(get_batch_fn, microbatch_sizes, optimizer)
            keys = list(self.ordered_modules.keys())[::-1]
            for name in keys:
                if self.ordered_modules[name] is None:
                    path = name.split(".")
                    modules = self.model._modules
                    for i in range(len(path) - 1):
                        modules = modules[path[i]]._modules
                    modules[path[-1]] = None
                    modules[path[-1]] = self.orig_modules[name]
                self.ordered_modules[name] = self.orig_modules[name]
                if isinstance(self.ordered_modules[name], CutPoint):
                    cutpoint = self.ordered_modules[name]
                    cutpoint.cp_index = -1
                    cutpoint.set_ret_val_func = None
                    cutpoint.stage = -1
                    cutpoint.device = None
                    cutpoint.send_fn = None
                    cutpoint.cp_func = None
            self.model.to(torch.float32)


    def dry_run(self, dummy_inputs, from_cache):
        # executes the forward pass of the module on dummy inputs. 
        # Sets the order in which modules are used and the total number of cutpoints declared.

        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        if self.local_rank == 0 and (not (from_cache and os.path.exists("_tmp_ord_mod") and os.path.exists("_tmp_inp_shapes"))):

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
            dist.barrier()

        else:
            dist.barrier()

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

        print("USED MODULES")
        for m in is_used:
            if not is_used[m]:
                path = m.split(".")
                modules = self.model._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                modules[path[-1]] = None
                modules[path[-1]] = PassThroughModule()
                self.ordered_modules[m] = None
            else:
                print(m)
    
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
        oom= False
        while (count < 50) and not oom:
            batch_size = 1
            warmup_time = time.time()
            try:
                inputs = get_batch_fn(batch_size, self.device)
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
                    break
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
            # print("warming up", warmup_time)
            count += 1

    def spawn_comm_workers(self, mBS):
        self.acts_recv_times = []; self.grads_recv_times = []
        self.acts_send_times = []; self.grads_send_times = []
        self.acts_receive_thread = self.grads_receive_thread = None
        self.acts_send_thread = self.grads_send_thread = None

        prev_rank = self.rank - 1 if self.rank > 0 else None
        next_rank = self.rank + 1 if self.rank < dist.get_world_size() - 1 else None
        comm_shape = list(self.input_shapes[ list(self.input_shapes.keys())[0] ] )
        comm_shape[0] = mBS

        if prev_rank is not None:
            acts_receive_thread = Thread(target=receiver, args=(prev_rank, comm_shape, self.acts_recv_times))
            acts_receive_thread.daemon=True
            acts_receive_thread.start()

            grads_send_thread = Thread(target=sender, args=(prev_rank, comm_shape, self.grads_send_times))
            grads_send_thread.daemon=True
            grads_send_thread.start()

        if next_rank is not None:
            grads_receive_thread = Thread(target=receiver, args=(next_rank, comm_shape, self.grads_recv_times))
            grads_receive_thread.daemon=True
            grads_receive_thread.start()

            acts_send_thread = Thread(target=sender, args=(next_rank, comm_shape, self.acts_send_times))
            acts_send_thread.daemon=True
            acts_send_thread.start()

    def end_comm_workers(self, mBS):
        if self.acts_receive_thread is not None:
            self.acts_receive_thread.join()
        if self.grads_receive_thread is not None:
            self.grads_receive_thread.join()
        if self.acts_send_thread is not None:
            self.acts_send_thread.join()
        if self.grads_send_thread is not None:
            self.grads_send_thread.join()

        acts_recv_times = self.acts_recv_times[5:]
        acts_send_times = self.acts_send_times[5:]
        grads_recv_times = self.grads_recv_times[5:]
        grads_send_times = self.grads_send_times[5:]

        act_send_time = act_recv_time = 0
        grad_send_time = grad_recv_time = 0

        if len(self.acts_send_times) > 0:
            act_send_time =  sum(acts_send_times)/len(acts_send_times)
        # if len(acts_recv_times) > 0:
        #     act_recv_time =  sum(acts_recv_times)/len(acts_recv_times)
        if len(self.grads_send_times) > 0:
            grad_send_time =  sum(grads_send_times)/len(grads_send_times)
        # if len(grads_recv_times) > 0:
        #     grad_recv_time =  sum(grads_recv_times)/len(grads_recv_times)

        gpus_per_node = 4
        if act_send_time > 0:
            long_send = (self.rank//gpus_per_node) != ((self.rank + 1)//gpus_per_node)
            self.comm_profile[mBS] = {"send": -1 if long_send else act_send_time,
                                   "long_send": act_send_time if long_send else -1}
        if grad_send_time > 0:
            long_send = (self.rank//gpus_per_node) != ((self.rank - 1)//gpus_per_node)
            self.comm_profile[mBS] = {"send": -1 if long_send else grad_send_time,
                                   "long_send": grad_send_time if long_send else -1}

        
    def profile(self, get_batch_fn,  microbatch_sizes, optimizer):

        if self.stage is not None:
            self.warmup(get_batch_fn, microbatch_sizes, optimizer)
        dist.barrier()
        self.compute_profile = {}
        self.comm_profile = {}
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        for batch_size in microbatch_sizes:
            self.model.train()
            self.spawn_comm_workers(batch_size)
            if self.stage is not None:
                try:
                    fwd_time, bwd_time, copy_time = self.profile_mbs(batch_size, get_batch_fn, optimizer)
                    self.compute_profile[batch_size] = {"fwd": fwd_time, "bwd": bwd_time, "copy": copy_time}
                    print(batch_size, fwd_time, bwd_time, copy_time)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("Out of memorryyyy")
                        break
                    else:
                        raise e
            self.end_comm_workers(batch_size)
            dist.barrier()

    def profile_fwd(self, inputs, batch_size):
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
        # _tosend = fwd_out.cpu()
        torch.cuda.synchronize(self.device)
        fwd_time = time.time() - start
        return fwd_out, fwd_time

    def profile_bwd(self, fwd_out, batch_size, optimizer):
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
        torch.cuda.synchronize(self.device)
        bwd_time = time.time() - bwd_time
        return bwd_time

    def profile_mbs(self, batch_size, get_batch_fn, optimizer):
        fwd_times = []
        bwd_times = []
        copy_times = []
        avg_mem_usage = 0
        for i in range(num_compute_passes): 
            torch.cuda.reset_max_memory_allocated(self.device)

            # get_batch_fn should load inputs into device and return dict
            input_mem = torch.cuda.memory_allocated(self.device)
            inputs = get_batch_fn(batch_size, device=self.device)
            input_mem = torch.cuda.memory_allocated(self.device) - input_mem

            fwd_out, fwd_time = self.profile_fwd(inputs, batch_size)

            bwd_time = self.profile_bwd(fwd_out, batch_size, optimizer) 

            copy_time = time.time()
            fwd_out.cpu()
            torch.cuda.synchronize()
            copy_time = time.time() - copy_time

            optimizer.step()
            for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
            for param in self.model.parameters():
                param.grad = None
                    
            fwd_act_size = fwd_out.element_size() * fwd_out.nelement()
            # del grads, fwd_out
            self.model.zero_grad()
            optimizer.zero_grad()

            mem_usage = torch.cuda.max_memory_allocated(self.device)
        
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
            copy_times.append(copy_time)
            avg_mem_usage += mem_usage

            del inputs
            self.fwd_inp = None; self.bwd_grad = None
        
            # opt_state_mem = torch.cuda.memory_allocated(self.device)
            # optimizer.state = collections.defaultdict(dict) # Reset state
            # opt_state_mem = opt_state_mem - torch.cuda.memory_allocated(self.device)
                
        # print("fwd:", fwd_times, 'bwd:', bwd_times)
        fwd_times = remove_outliers(fwd_times)
        bwd_times = remove_outliers(bwd_times)
        fwd_time = sum(fwd_times)/len(fwd_times)
        bwd_time = sum(bwd_times)/len(bwd_times)
        copy_time = sum(copy_times)/len(copy_times)
        mem_usage = avg_mem_usage / num_compute_passes
        return fwd_time, bwd_time, copy_time
        
        
                
class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None

