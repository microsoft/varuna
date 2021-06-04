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
num_comm_passes = 30

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

def receiver(recv_rank, recv_shape, recv_times, dtype):
    chunks = num_comm_passes
    recv_handles = Queue()

    for i in range(chunks):
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

    del acts_tensor
    
def sender(send_rank, send_shape, send_times, dtype):
    chunks = num_comm_passes
    send_handles = Queue()

    for i in range(chunks):
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

    del output_acts

class Profiler:

    def __init__(self, model, device, gpus_per_node=None, fp16 = False):
        self.model = model
        self.ret_val = None
        self.fp16 = fp16
        self.device = device
        torch.cuda.set_device(device)

        self.rank = dist.get_rank()
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        if gpus_per_node is None:
            gpus_per_node = torch.cuda.device_count()
        self.gpus_per_node = gpus_per_node
        self.world_size = dist.get_world_size()
        self.pre_cp = None
        self.warmed_up = False
        self.comm_profile = {}

    def initialize(self, dummy_inputs, stages_to_profile=None,from_cache=True):
        self.dry_run(dummy_inputs, from_cache)
        if stages_to_profile is None:
            stages_to_profile = range(self.num_cutpoints+1)
        my_stages_to_profile = \
            list(range(self.rank, len(stages_to_profile), self.world_size))
        my_stages_to_profile = [stages_to_profile[i] for i in my_stages_to_profile]
        self.prev_stages_to_profile = [-1 for _ in my_stages_to_profile]
        if self.rank > 0:
            i_ = 0
            for i in range(self.rank-1, len(stages_to_profile), self.world_size):
                if i_ >= len(my_stages_to_profile):
                    break
                self.prev_stages_to_profile[i_] = stages_to_profile[i]
                i_ += 1
        self.next_stages_to_profile = [-1 for _ in my_stages_to_profile]
        if self.rank < self.world_size - 1:
            i_ = 0
            for i in range(self.rank+1, len(stages_to_profile), self.world_size):
                if i_ >= len(my_stages_to_profile):
                    break
                self.next_stages_to_profile[i_] = stages_to_profile[i]
                i_ += 1

        self.num_rounds = math.ceil(len(stages_to_profile) / self.world_size)
        print("total rounds", self.num_rounds)
        print("stages", my_stages_to_profile)
        print("prev stages", self.prev_stages_to_profile)
        print("next stages", self.next_stages_to_profile, flush=True)

        self.stages_to_profile = my_stages_to_profile

        self.orig_modules = dict()
        for name in self.ordered_modules:
            self.orig_modules[name] = self.ordered_modules[name]

        self.dummy_inputs = dummy_inputs

    def profile_all(self, get_batch_fn,  microbatch_sizes, get_optimizer_fn, out_folder="profiles" ):

        factors, alr_sizes = self.get_all_reduce_sizes()
        optimizer = None
        for i in range(self.num_rounds):
            self.stage = None
            self.prev_stage = -1; self.next_stage = -1
            if i < len(self.stages_to_profile):
                self.stage = self.stages_to_profile[i]
                self.prev_stage = self.prev_stages_to_profile[i]
                self.next_stage = self.next_stages_to_profile[i]
                print("STAGE", self.stage)
                self.trim_model(self.stage, self.stage + 1)
                self.check_unused_parameters(self.dummy_inputs)
                self.model.to(self.device)
                optimizer = get_optimizer_fn(self.model)
                

            # print("pre-profile mem",torch.cuda.memory_allocated(self.device))
            self.profile(get_batch_fn, microbatch_sizes, optimizer)
            
            if self.stage is not None:
                optimizer.state = collections.defaultdict(dict) # Reset state
                with open(os.path.join(out_folder, f"compute-profile-{self.stage}"), "wb") as f:
                    pickle.dump(self.compute_profile, f)
                
            keys = self.comm_profile.keys()
            # print("Communication")
            # for mbs in keys:
            #     print(mbs, self.comm_profile[mbs]["send"], self.comm_profile[mbs]["long_send"])
            # print("---------"*5)


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

            for n in self.orig_params:
                p = self.orig_params[n]
                path = n.split(".")
                parent = self.model
                for i in range(len(path) - 1):
                    parent = getattr(parent, path[i])
                setattr(parent,path[-1], p)
            self.model = self.model.to('cpu')

        print("pre-alr mem",torch.cuda.memory_allocated(self.device))
        self.profile_all_reduce(factors, alr_sizes)
        if self.rank == 0:
            with open(os.path.join(out_folder, "allred-profile"), "wb") as f:
                pickle.dump(self.all_reduce_profile, f)

        # gather comm profile
        obj_list = [None for _ in range(dist.get_world_size())] if self.rank == 0 else None
        dist.gather_object(self.comm_profile, object_gather_list = obj_list, dst=0)
        if self.rank == 0:
            aggregate_comm_profile = dict()
            for comm_profile in obj_list:
                for comm_shape in comm_profile:
                    if comm_shape not in aggregate_comm_profile:
                        aggregate_comm_profile[comm_shape] = {"send": [], "long_send": []}
                    
                    full_profile = aggregate_comm_profile[comm_shape]
                    this_profile = comm_profile[comm_shape]
                    full_profile["send"].extend(this_profile["send"])
                    full_profile["long_send"].extend(this_profile["long_send"])

            for comm_shape in aggregate_comm_profile:
                for key in ["send","long_send"]:
                    comm_times = aggregate_comm_profile[comm_shape][key]
                    if len(comm_times) > 0:
                        avg_time = sum(comm_times)/len(comm_times)
                        aggregate_comm_profile[comm_shape][key] = avg_time * 1000000
                        print(f"{comm_shape} {key}: {avg_time}")
                    else:
                        print(f"WARNING: No comm times for size {comm_shape} {key}!")
                        aggregate_comm_profile[comm_shape][key] = -1

            with open(os.path.join(out_folder, "comm-profile"), "wb") as f:
                pickle.dump(aggregate_comm_profile, f)

        print("All reduce times")
        for f in factors:
            print(f, self.all_reduce_profile[f])

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
            # else:
            #     print(m)
        
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
        self.orig_params = dict()
        for n,p in self.model.named_parameters():
            if p.grad is None:
                to_remove.append(n)
                path = n.split(".")
                parent = self.model
                for i in range(len(path) - 1):
                    parent = getattr(parent, path[i])
                self.orig_params[n] = p
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
    
    def get_all_reduce_sizes(self):
        
        factors = []
        num_pstages = self.num_cutpoints + 1
        for i in range(1,num_pstages + 1):
            if num_pstages % i == 0:
                factors += [i]
        # factors = set(factors)

        modules = self.ordered_modules
        index = 1
        pstage_to_param_count = dict()
        pstage_to_param_count[0] = 0

        for name in modules:
            if name == "":
                continue
            module = modules[name]

            pstage_param_count = 0
            for p in module.parameters(recurse = False):
                pstage_param_count += p.numel()
            pstage_to_param_count[index-1] += pstage_param_count
            
            # only need to set up two cutpoints at most
            if isinstance(module, CutPoint): 
                pstage_to_param_count[index] = 0  
                index += 1

        factors = sorted(factors)[::-1]
        param_sizes = []
        for f in factors:
            param_size = 0
            num_pstages_per_stage = num_pstages // f
            for i in range(num_pstages_per_stage):
                param_size += pstage_to_param_count[i]
            param_sizes.append(param_size)

        print("factors", factors)
        print("all reduce sizes", param_sizes)

        return factors, param_sizes
            

    def set_ret_val(self, val):
        self.ret_val = val

    def recv(self, grads=False):
        if grads:
            return self.bwd_grad
        return self.fwd_inp

    def warmup(self, get_batch_fn, microbatch_sizes, optimizer):
        
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        count = 0
        while count < 50:
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
                fwd_out.cpu()
                grads = 0.00001 * torch.ones(list(fwd_out.size()), device = self.device)
                if self.bwd_grad_shape is not None:
                    self.bwd_grad = torch.ones(self.bwd_grad_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)

                # print("grads before",optimizer.param_groups[0]["params"][0].grad)
                if self.fp16:
                    with amp.scale_loss(fwd_out, optimizer, last_partition=True) as scaled_out:
                        scaled_out.backward(grads)
                else:
                    fwd_out.backward(grads)
                del grads, fwd_out

            except RuntimeError as e:
                grads = None; fwd_out = None
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
            self.model.zero_grad()
            optimizer.zero_grad()
            self.ret_val = None
            self.fwd_inp = None; self.bwd_grad = None

            warmup_time = time.time() - warmup_time
            count += 1

        if self.pre_cp is not None:
            self.pre_cp.recv_fn = None

        optimizer.state = collections.defaultdict(dict) # Reset state

    def spawn_comm_workers(self, mBS):
        self.acts_recv_times = []; self.grads_recv_times = []
        self.acts_send_times = []; self.grads_send_times = []
        self.acts_receive_thread = self.grads_receive_thread = None
        self.acts_send_thread = self.grads_send_thread = None

        prev_rank = self.rank - 1 if self.rank > 0 else None
        next_rank = self.rank + 1 if self.rank < dist.get_world_size() - 1 else None
        dtype = torch.float16 if self.fp16 else torch.float32

        if prev_rank is not None and (self.prev_stage not in [-1,self.num_cutpoints]):
            prev_cp_name = list(self.input_shapes.keys())[self.prev_stage]
            comm_shape = list(self.input_shapes[prev_cp_name][0])
            comm_shape[0] = mBS
            comm_size = 1
            for d in comm_shape:
                comm_size *= d
            if comm_size not in self.comm_profile:
                self.comm_profile[comm_size] = {"send": [], "long_send": []}
            self.prev_rank_comm_shape = comm_size
            self.acts_receive_thread = Thread(target=receiver, args=(prev_rank, comm_shape, self.acts_recv_times, dtype))
            self.acts_receive_thread.daemon=True
            self.acts_receive_thread.start()

            self.grads_send_thread = Thread(target=sender, args=(prev_rank, comm_shape, self.grads_send_times, dtype))
            self.grads_send_thread.daemon=True
            self.grads_send_thread.start()

        if self.stage is not None and self.next_stage != -1:
            comm_shape = list(self.bwd_grad_shape)
            comm_shape[0] = mBS
            comm_size = 1
            for d in comm_shape:
                comm_size *= d
            if comm_size not in self.comm_profile:
                self.comm_profile[comm_size] = {"send": [], "long_send": []}
            self.next_rank_comm_shape = comm_size
            self.grads_receive_thread = Thread(target=receiver, args=(next_rank, comm_shape, self.grads_recv_times, dtype))
            self.grads_receive_thread.daemon=True
            self.grads_receive_thread.start()

            self.acts_send_thread = Thread(target=sender, args=(next_rank, comm_shape, self.acts_send_times, dtype))
            self.acts_send_thread.daemon=True
            self.acts_send_thread.start()

    def end_comm_workers(self, mBS):
        if self.acts_receive_thread is not None:
            self.acts_receive_thread.join()
        if self.grads_receive_thread is not None:
            self.grads_receive_thread.join()
        if self.acts_send_thread is not None:
            self.acts_send_thread.join()
        if self.grads_send_thread is not None:
            self.grads_send_thread.join()


        gpus_per_node = self.gpus_per_node
        if len(self.acts_send_times) > 0:
            comm_size = self.next_rank_comm_shape
            long_send = (self.rank//gpus_per_node) != ((self.rank + 1)//gpus_per_node)
            send_times = self.comm_profile[comm_size]["long_send"] if long_send else \
                        self.comm_profile[comm_size]["send"]
            send_times.extend(self.acts_send_times)

        if len(self.grads_send_times) > 0:
            comm_size = self.prev_rank_comm_shape
            long_send = (self.rank//gpus_per_node) != ((self.rank - 1)//gpus_per_node)
            send_times = self.comm_profile[comm_size]["long_send"] if long_send else \
                        self.comm_profile[comm_size]["send"]
            send_times.extend(self.grads_send_times)


    def profile_all_reduce(self, factors, alr_sizes):
        num_passes = 3
        self.all_reduce_profile = dict()

        # init with alr time for ring 1
        for f in factors:
            self.all_reduce_profile[f] = [0.0]

        for ring_size in range(2,self.world_size+1):
            # TODO: all-reduce groups in clustered mode for gpus_per_vm > 1 and acc to dp_size possibilities
            #       and ring sizes only powers of 2
            ranks = list(range(ring_size))
            group = torch.distributed.new_group(ranks=ranks, backend='nccl')
            if dist.get_rank() < ring_size:
                oom = torch.cuda.IntTensor([0])
                for factor, alr_size in zip(factors, alr_sizes):
                    print(f"alr {factor} {alr_sizes} {ring_size}")
                    try:
                        allred_tensor = torch.ones(alr_size, dtype=torch.float16 if self.fp16 else torch.float32, 
                                device=self.device)
                        avg_time = 0.0
                        for _ in range(num_passes):
                            allred_time = time.time()
                            dist.all_reduce(allred_tensor, group=group)
                            torch.cuda.synchronize()
                            allred_time = time.time() - allred_time
                            allred_tensor /= ring_size
                            avg_time += allred_time
                        avg_time /= num_passes

                        avg_time = torch.tensor([avg_time * 1000], device=self.device) / ring_size
                        dist.all_reduce(avg_time, group = group)
                        avg_time = avg_time.item() * 1000
                        self.all_reduce_profile[factor].append(avg_time)            

                    except RuntimeError as e:
                        allred_tensor = None
                        if 'out of memory' in str(e):
                            print("Out of memorryyyy")
                            oom = torch.cuda.IntTensor([1])
                        else:
                            raise e
                    dist.all_reduce(oom, group=group)
                    if oom.item():
                        break

    def profile(self, get_batch_fn,  microbatch_sizes, optimizer):

        if self.stage is not None and not self.warmed_up:
            self.warmup(get_batch_fn, microbatch_sizes, optimizer)
            self.warmed_up = True
        
        dist.barrier()

        self.compute_profile = {}
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = self.recv

        oom = torch.cuda.IntTensor([0])
        for batch_size in microbatch_sizes:
            self.model.train()
            self.spawn_comm_workers(batch_size)
            if self.stage is not None:
                # print("pre-mbs mem",torch.cuda.memory_allocated(self.device))
                try:
                    fwd_time, bwd_time, copy_time, mem_usage, fwd_act_size = \
                        self.profile_mbs(batch_size, get_batch_fn, optimizer)
                    fwd_time *= 1000000; bwd_time *= 1000000; copy_time *= 1000000
                    self.compute_profile[batch_size] = {"fwd": fwd_time, "bwd": bwd_time, \
                                    "copy": copy_time,"max_memory": mem_usage, "acts_size": fwd_act_size }
                    print(batch_size, fwd_time, bwd_time, copy_time, mem_usage, fwd_act_size)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("Out of memorryyyy")
                        oom = torch.cuda.IntTensor([1])
                    else:
                        raise e
            self.end_comm_workers(batch_size)
            dist.all_reduce(oom)
            if oom.item():
                break

    def profile_fwd(self, inputs, batch_size):
        if self.fwd_inp_shape is not None:
            fwd_inp_shape = list(self.fwd_inp_shape)
            fwd_inp_shape[0] = batch_size
            self.fwd_inp = torch.ones(fwd_inp_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
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
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_end.record()
        return fwd_out, fwd_start, fwd_end

    def profile_bwd(self, fwd_out, batch_size, optimizer):
        if isinstance(fwd_out, tuple):
            fwd_out = fwd_out[0]
        grads = 0.00001 * torch.ones(list(fwd_out.size()), device = self.device)
        if self.bwd_grad_shape is not None:
            self.bwd_grad = torch.ones(self.bwd_grad_shape, dtype = torch.float16 if self.fp16 else torch.float32).to(self.device)
    
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_start.record()
        if self.fp16:
            with amp.scale_loss(fwd_out, optimizer, last_partition=True) as scaled_out:
                scaled_out.backward(grads)
        else:
            fwd_out.backward(grads)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end.record()
        del grads
        return fwd_out, bwd_start, bwd_end

    def profile_mbs(self, batch_size, get_batch_fn, optimizer):
        fwd_times = []
        bwd_times = []
        copy_times = []
        avg_mem_usage = 0
        # copy_tensor = None
        for i in range(num_compute_passes): 
            torch.cuda.reset_max_memory_allocated(self.device)

            # get_batch_fn should load inputs into device and return dict
            input_mem = torch.cuda.memory_allocated(self.device)
            inputs = get_batch_fn(batch_size, device=self.device)
            input_mem = torch.cuda.memory_allocated(self.device) - input_mem

            fwd_out, fwd_start, fwd_end = self.profile_fwd(inputs, batch_size)

            fwd_out, bwd_start, bwd_end = self.profile_bwd(fwd_out, batch_size, optimizer) 

            # if copy_tensor is None:
            #     copy_tensor = torch.ones_like(fwd_out, device=self.device)

            optimizer.step()
            for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
            for param in self.model.parameters():
                param.grad = None
                    
            fwd_act_size = fwd_out.element_size() * fwd_out.nelement()
            self.model.zero_grad()
            optimizer.zero_grad()

            torch.cuda.synchronize(self.device)
            fwd_time = fwd_start.elapsed_time(fwd_end) / 1000
            bwd_time = bwd_start.elapsed_time(bwd_end) / 1000

            copy_start = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            with torch.no_grad():
                copy_time = time.time()
                _t = fwd_out.cpu()
                copy_time = time.time() - copy_time
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_end.record()

            mem_usage = torch.cuda.max_memory_allocated(self.device)
        
            torch.cuda.synchronize(self.device)
            copy_time = copy_start.elapsed_time(copy_end) / 1000
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
            copy_times.append(copy_time)
            avg_mem_usage += mem_usage

            del inputs, fwd_out
            self.fwd_inp = None; self.bwd_grad = None
    
        opt_state_mem = torch.cuda.memory_allocated(self.device)
        optimizer.state = collections.defaultdict(dict) # Reset state
        opt_state_mem = opt_state_mem - torch.cuda.memory_allocated(self.device)
                
        # print("copy:", copy_tensor.size() ,copy_times)
        fwd_times = remove_outliers(fwd_times)
        bwd_times = remove_outliers(bwd_times)
        copy_times = remove_outliers(copy_times)
        fwd_time = sum(fwd_times)/len(fwd_times)
        bwd_time = sum(bwd_times)/len(bwd_times)
        copy_time = sum(copy_times)/len(copy_times)
        mem_usage = avg_mem_usage / num_compute_passes
        return fwd_time, bwd_time, copy_time, mem_usage, fwd_act_size
        
        
                
class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None

