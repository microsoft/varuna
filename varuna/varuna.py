from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.multiprocessing import Process
from queue import Queue
from threading import Thread
import math
from apex import amp
import time
from apex.amp import _amp_state
import amp_C, apex_C
from apex.multi_tensor_apply import multi_tensor_applier
import concurrent.futures
import shutil

from .partitioned_model import PartitionedModel
import gc
import numpy
# from hashlib import sha1
import socket

import os
import sys
import time

Module = nn.Module

log_verbose = False

TASK = ["fwd", "rec", "bwd"]

def share_weight_grads(model, tied_group):
    parameter_names = model.parameter_names
    rank_within_stage = model.stage_to_rank_map[model.stage].index(model.rank)
    for i,w in enumerate(model.shared_weights):
        recv_stage, send_stage = model.shared_weight_stages[i]
        if recv_stage == send_stage:
            continue
        if model.stage == send_stage:
            for p in parameter_names:
                if parameter_names[p] == w[1]:
                    send_weight = p
                    break
            dist.all_reduce(send_weight.grad.data, group=tied_group)
        elif model.stage == recv_stage:
            for p in parameter_names:
                if parameter_names[p] == w[0]:
                    recv_weight = p
                    break
            dist.all_reduce(recv_weight.grad.data, group=tied_group)

    
class Varuna(Module):
    """
    model = nn.Sequential(a,b,c,d)
    model = Varuna(model, microbatches/minibatch, list_of_devices)
    for iteration in epoch:
        model(input)   # execute Varuna's pipeline (forward and backward pass)
        optimizer.step()
        optimizer.zero_grad()
    """
    def __init__(self,
                model,
                stage_to_rank_map,
                dummy_inputs,
                batch_size,
                chunk_size,
                fp16 = False, 
                local_rank=-1,
                device=-1,
                shared_weights=None):
        super().__init__()

        self.partitions = len(stage_to_rank_map)
        self.rank = dist.get_rank()
        self.local_rank = local_rank if local_rank != -1 else self.rank
        self.stage_to_rank_map = stage_to_rank_map

        self.stage = -1
        for stage in self.stage_to_rank_map:
            i = 0
            for rank in self.stage_to_rank_map[stage]:
                if rank == self.rank:
                    rank_within_stage = i
                    data_depth = len(self.stage_to_rank_map[stage])
                    self.stage = stage
                    break
                i += 1
        if self.stage == -1:
            raise ValueError("Rank " + str(self.rank) + " not found in stage to rank map!")
        self.data_parallel = data_depth > 1

        if device == -1:
            device = self.local_rank
        torch.cuda.set_device(device)
        self.device = torch.device("cuda", device)

        self.optimizer = None
        self.fp16 = fp16
        self.shared_weights = shared_weights

        if self.data_parallel and not self.fp16:
            raise RuntimeError("Data parallel for fp32 is currently broken")

        # partition model based on "CutPoint"s using a dry run with dummy inputs (dict)
        self.model = PartitionedModel(model, self.rank, self.local_rank, device, self.stage_to_rank_map, self.fp16, shared_weights)
        self.model.initialize( dummy_inputs, from_cache=False )
        if self.local_rank==0:
            # print("varuna init() after dry run init: ", self.local_rank, torch.cuda.memory_summary(self.device))
            print('varuna init() after dry run:', self.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        self.partitioned_model = self.model
        self.shared_weight_stages = self.model.shared_weight_stages if self.shared_weights is not None else None

        print("SHARED WEIGHTS ARE")
        print(self.shared_weight_stages)

        self.batch_size = batch_size // data_depth
        self.micro_batch_size = chunk_size
        self.last_chunk_size = self.batch_size % chunk_size 
        self.init_communication(rank_within_stage)

        self.model.to(self.device)        
        self.init_distributed()

        self.config = {
            "stage": self.stage,
            "partitions": self.partitions,
            "fp16": self.fp16,
            "fwd_inp_shape": self.fwd_inp_shape,
            "bwd_grad_shape": self.bwd_grad_shape,
            "receive_rank": self.receive_rank,
            "send_rank": self.send_rank,
            "device": self.device,
            "data_depth": len(self.stage_to_rank_map[self.stage]),
            "dp_process_group": self.process_group, 
            "pipeline_process_group": self.pipeline_group,
            "tied_group": self.tied_group,
            "make_logfile": False, #bool(self.rank == self.stage_to_rank_map[self.stage][-1]),
            "last_chunk_size": self.last_chunk_size,
            "shared_weights": self.shared_weights,
            "shared_weight_stages": self.shared_weight_stages,
            "stage_to_rank_map": self.stage_to_rank_map,
            "local_rank": self.local_rank,
            "chunk_size": chunk_size,
            "rank_within_stage": stage_to_rank_map[self.stage].index(self.rank)
        }

        self.schedule = self.generate_schedule()
        self.step = 0

    def init_communication(self, rank_within_stage):
        self.send_rank = None; self.receive_rank = None

        # send ranks
        if self.stage < (self.partitions-1):
            self.send_rank = self.stage_to_rank_map[self.stage + 1][rank_within_stage]

        # receive ranks
        if self.stage > 0:
            self.receive_rank = self.stage_to_rank_map[self.stage - 1][rank_within_stage]

        # set expected shapes of inputs and gradients for each partition
        self.fwd_inp_shape = self.bwd_grad_shape = None
        if self.stage > 0:
            self.fwd_inp_shape = self.model.forward_input_shapes[0]
            self.fwd_inp_shape[0] = self.micro_batch_size
            # print("Varuna fwd inp shape ", self.fwd_inp_shape)
        if self.stage < (self.partitions-1):
            self.bwd_grad_shape = self.model.backward_grad_shapes[0]
            self.bwd_grad_shape[0] = self.micro_batch_size
            # print("Varuna bwd grad shape", self.bwd_grad_shape)

    def init_distributed(self):
        # create same process groups on all ranks
        self.process_group = None
        process_groups = {}
        for stage in range(self.partitions):
            ranks = self.stage_to_rank_map[stage]
            if len(ranks) > 1:
                process_groups[stage] = dist.new_group(ranks=ranks,backend='nccl')
            else:
                process_groups[stage] = None

        if process_groups[self.stage] is not None:
            self.process_group = process_groups[self.stage]

        # get stream to rank map
        depth = len(self.stage_to_rank_map[self.stage])
        world_size = depth * self.partitions

        stream_to_rank_map = {}
        for i in range(depth):
            stream = []
            for stage in range(self.partitions):
                stream.append(self.stage_to_rank_map[stage][i])
            stream_to_rank_map[i] = stream

        self.tied_group = None
        self.pipeline_group = None
        pipeline_groups = {}
        tied_groups = {}
        for stream in range(depth):
            ranks = stream_to_rank_map[stream]
            if len(ranks) > 1:
                pipeline_groups[stream] = dist.new_group(ranks=ranks)
                recv_stage, send_stage = self.shared_weight_stages[0]
                tied_ranks = [ranks[recv_stage], ranks[send_stage]]
                tied_groups[stream] = dist.new_group(ranks=tied_ranks)
            else:
                pipeline_groups[stream] = None
                tied_groups[stream] = None
            
        current_stream = self.stage_to_rank_map[self.stage].index(self.rank)
        print("this rank ", self.rank, "is part of pipeline stream ", current_stream)
        if pipeline_groups[current_stream] is not None:
            self.pipeline_group = pipeline_groups[current_stream]
            self.tied_group = tied_groups[current_stream]

    def forward(self, inputs):
        if self.fp16:
            assert self.optimizer is not None, "For fp16, you must set the optimizer using set_optimizer()"        
        
        # Divide a mini-batch into micro-batches.
        batches = scatter(inputs, int(self.batch_size),self.micro_batch_size)
        
        # need not pass the first argument if rank!=0
        # avoid dataloader compute in machines other than the first
        # ask the model writer to pass the input batch generating dataloader function to Varuna::__init__
        # and Varuna can take care of input dataloader explicitly
        self.config["make_logfile"] = bool(self.config["make_logfile"] and self.step<120)
        self.config["parameter_names"] = self.parameter_names
        batch_time = time.time()
        pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer)
        loss, overflow, global_grad_norm = pipeline.run()
        batch_time = time.time() - batch_time
        self.step += 1

        if self.rank == 0 and self.step%10==0:
            manager_ip = "10.0.3.4"
            manager_port = 5000
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    message = "progress {}".format(self.step)
                    sock.connect((manager_ip, manager_port))
                    sock.sendall(bytes(message, 'ascii'))
            except:
                print("Could not send progress update message")
        
        return loss, overflow, global_grad_norm

    def evaluate(self, inputs):
        assert isinstance(inputs, dict), "input must be a dictionary!"

        # self.partitioned_model.eval()
        def send(x, grads=False):
            # print("sending to rank", self.send_rank, x.size())
            dist.send(x.cpu(), self.send_rank)
        def recv(grads=False):
            x_shape = self.fwd_inp_shape
            x = torch.zeros(x_shape, dtype=torch.float16 if self.fp16 else torch.float32)
            # print("receiving from rank", self.receive_rank, x_shape)
            dist.recv(x, self.receive_rank)
            return x.to(self.device)
        self.partitioned_model.set_send_fn(send)
        self.partitioned_model.set_recv_fn(recv)

        batches = scatter(inputs, int(self.batch_size),self.micro_batch_size)
        
        with torch.no_grad():
            avg_output = None
            for mb in batches[:-1]:
                output = self.partitioned_model(**mb)
                avg_output = output if avg_output is None else avg_output + output
            mb = batches[-1]
            def recv(grads=False):
                x_shape = list(self.fwd_inp_shape)
                if self.last_chunk_size > 0:
                    x_shape[0] = self.last_chunk_size
                x = torch.zeros(x_shape, dtype=torch.float16 if self.fp16 else torch.float32)
                # print("last receiving from rank", self.receive_rank, x_shape)
                dist.recv(x, self.receive_rank)
                return x.to(self.device)
            self.partitioned_model.set_recv_fn(recv)
            output = self.partitioned_model(**mb)
            if self.stage == self.partitions - 1:
                avg_output = output if avg_output is None else avg_output + output

        if self.stage == self.partitions - 1:
            avg_output /= len(batches)
        return avg_output

    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.model.zero_grad()
    
    def checkpoint(self, cp_dir_name):
        return self.partitioned_model.checkpoint(cp_dir_name)

    def checkpoint_optimizer(self, optimizer, parameter_to_name, param_name_to_pstage, \
                                cp_dir_name, tempdir=None, on_demand = False, shard=False):
        cp_time = time.time()
        mv_futures = []
        if tempdir is not None:
            executor = concurrent.futures.ThreadPoolExecutor()

        rank_within_stage = self.stage_to_rank_map[self.stage].index(self.rank)
        depth = len(self.stage_to_rank_map[self.stage]) if shard else 1

        # shard checkpoint over DP workers
        if rank_within_stage == 0 or shard:
            cuts_per_stage = self.partitioned_model.cuts_per_stage
            # save param states for each cutpoint separately
            pstages = range(cuts_per_stage * self.stage, (self.stage+1)* cuts_per_stage)
            pstage_state_dicts = dict()
            for i in pstages:
                pstage_state_dicts[i] = dict()

            ind = 0
            for key in optimizer.state:
                # for sharding
                if ind % depth != rank_within_stage:
                    ind += 1
                    continue
                # store state by param names instead of actual parameters
                param_name = parameter_to_name[key]
                assert param_name in param_name_to_pstage, "param {} not found in rank {}".format(param_name,dist.get_rank())
                pstage = param_name_to_pstage[param_name]
                pstage_state_dicts[pstage][param_name] = optimizer.state[key]
                ind += 1
                
            if tempdir is not None:
                for i in pstages:
                    temp_name =  os.path.join(tempdir,"opt-state-" + str(i))
                    cp_name = os.path.join(cp_dir_name,"opt-state-" + str(i))
                    if depth > 1:
                        temp_name += "_" + str(rank_within_stage)
                        cp_name += "_" + str(rank_within_stage)
                    torch.save(pstage_state_dicts[i], temp_name)
                    mv_futures.append(executor.submit(shutil.move, temp_name, cp_name))
            else:
                for i in pstages:
                    cp_name = os.path.join(cp_dir_name,"opt-state-" + str(i))
                    if depth > 1:
                        cp_name += "_" + str(rank_within_stage)
                    torch.save(pstage_state_dicts[i], cp_name)

            # also store optimizer master params for mixed precision training
            if self.fp16:

                pstage_state_dicts = dict()
                for i in pstages:
                    pstage_state_dicts[i] = dict()

                ind = 0
                for p in amp.master_params(optimizer):
                    if ind % depth != rank_within_stage:
                        ind += 1
                        continue
                    param_name = parameter_to_name[p]
                    # not a part of the worker's stage
                    if param_name not in param_name_to_pstage:
                        continue
                    pstage = param_name_to_pstage[param_name]
                    if pstage not in pstages:
                        continue
                    pstage_state_dicts[pstage][param_name] = p
                    ind += 1
                
                if tempdir is not None:
                    for i in pstages:
                        temp_name =  os.path.join(tempdir,"opt-fp32-params-" + str(i))
                        cp_name = os.path.join(cp_dir_name,"opt-fp32-params-" + str(i))
                        if depth > 1:
                            temp_name += "_" + str(rank_within_stage)
                            cp_name += "_" + str(rank_within_stage)
                        torch.save(pstage_state_dicts[i], temp_name)
                        mv_futures.append(executor.submit(shutil.move, temp_name, cp_name))
                else:
                    for i in pstages:
                        cp_name = os.path.join(cp_dir_name,"opt-fp32-params-" + str(i))
                        if depth > 1:
                            cp_name += "_" + str(rank_within_stage)
                        torch.save(pstage_state_dicts[i], cp_name)

        cp_time = time.time() - cp_time
        print("Opt ckpt time", cp_time)
        return mv_futures

    
    def to(self, device):
        self.model.to(device)
    
    def generate_schedule(self):
        chunks = math.ceil(self.batch_size / self.micro_batch_size)
        print(chunks,"chunks")
        c_schedule = os.popen(os.path.join(os.path.dirname(os.path.abspath(__file__)),'genschedule ')+str(self.partitions)+' '+str(chunks)+' '+str(self.stage)).read()
        schedule = list()
        steps = c_schedule.split(';')
        steps = steps[:-1]
        for step in steps:
            task = step.split(',')
            schedule.append((int(task[0]), int(task[1])))
        
        return schedule
                

def save_rng_states(device):
    """capture current CPU, GPU random number generator states to reuse while recomputing activations
    in order to ensure Referential Transparency
    """
    cpu_rng_state = torch.get_rng_state()

    gpu_rng_states: Optional[ByteTensor]
    # gpu_rng_states = torch.cuda.get_rng_state_all() 
    gpu_rng_states = torch.cuda.get_rng_state(device)
    return (cpu_rng_state, gpu_rng_states)

def restore_rng_states(rng_states, device):
    cpu_rng_state, gpu_rng_states = rng_states
    torch.set_rng_state(cpu_rng_state)
    # torch.cuda.set_rng_state_all(gpu_rng_states)        # todo: verify correctness;   batchNorm, dropouts, convlayers?
    torch.cuda.set_rng_state(gpu_rng_states, device)


class Pipeline:
    """ Pipeline parallelism for Varuna """

    def __init__(self, batches, model, config, schedule, optimizer):
        self.batches = batches
        self.partitions = config["partitions"]
        self.stage = config["stage"]
        self.data_depth = config["data_depth"]
        self.data_parallel = bool(self.data_depth > 1)
        self.process_group = config["dp_process_group"]
        self.pipeline_group = config["pipeline_process_group"]
        self.tied_group = config["tied_group"]
        self.rank_within_stage = config["rank_within_stage"]

        self.model = model
        self.partitioned_model = self.model#.module if self.data_parallel else self.model
        self.device = config["device"]
        self.schedule = schedule
        self.fp16 = config["fp16"]
        self.rank = dist.get_rank()

        self.fwd_inp_shape = config["fwd_inp_shape"]
        self.bwd_grad_shape = config["bwd_grad_shape"]
        self.parameter_names = config["parameter_names"]

        self.shared_weights = config["shared_weights"]
        self.shared_weight_stages = config["shared_weight_stages"]
        self.stage_to_rank_map = config["stage_to_rank_map"]
        self.local_rank = config["local_rank"]

        self.make_logfile = config["make_logfile"]
        if self.make_logfile:
            replica_num = self.stage_to_rank_map[self.stage].index(self.rank)
            microBS = config["chunk_size"]
            logfilename = "/mnt/default/projects/varuna/gantts/8pclustered11/varuna_logs-"+str(self.data_depth)+"dp-" + str(microBS) + "mBS-stage" + str(self.stage) + "of" + str(self.partitions) + "_" + str(replica_num)
            # logfilename = os.path.join("/home/varuna/gpt2-blob/perf_analysis_1.5b","stats",logfilename)
            self.logfile = open(logfilename,"a")
            self.logfile.write("start time {}\n".format(time.time()))

        self.receive_rank = config["receive_rank"]
        self.send_rank = config["send_rank"]

        self.last_chunk_size = config["last_chunk_size"]

        self.optimizer = optimizer
        self.fp16 = config["fp16"]

        self.grads_send_queue = Queue()
        self.acts_send_queue = Queue()
        self.spawn_send_workers()

        self.acts_queue = Queue()       # activation at the boundary, rename as input_acts
        self.grads_queue = Queue()
        self.recompute_queue = Queue()

        self.acts_receive_thread = None
        self.grads_receive_thread = None
        self.acts_send_thread = None
        self.grads_send_thread = None

        self.back_start_times = Queue()

        # stores output of recompute(/forward) pass to be used by backward()
        self.loss = None
        self.average_loss = 0

    def spawn_receive_workers(self):
        if self.stage > 0:
            self.acts_receive_thread = Thread(target=self.acts_receiver, args=())
            self.acts_receive_thread.daemon=True
            self.acts_receive_thread.start()

        if self.stage < self.partitions-1:
            self.grads_receive_thread = Thread(target=self.grads_receiver, args=())
            self.grads_receive_thread.daemon=True
            self.grads_receive_thread.start()
    
    def spawn_send_workers(self):
        if self.stage < self.partitions-1:
            self.acts_send_thread = Thread(target=self.acts_sender, args=())
            self.acts_send_thread.daemon=True
            self.acts_send_thread.start()

        if self.stage > 0:
            self.grads_send_thread = Thread(target=self.grads_sender, args=())
            self.grads_send_thread.daemon=True
            self.grads_send_thread.start() 
    
    def acts_receiver(self):
        chunks = len(self.batches)
        dtype = torch.float16 if self.fp16 else torch.float32
        recv_handles = Queue()

        for task,index in self.schedule:
            if task == 0:
                fwd_inp_shape = self.fwd_inp_shape
                if index == (chunks-1) and self.last_chunk_size > 0:
                    fwd_inp_shape = list(self.fwd_inp_shape)
                    fwd_inp_shape[0] = self.last_chunk_size
                acts_tensor = torch.ones(fwd_inp_shape, dtype=dtype)
                handle = dist.irecv(acts_tensor, src=self.receive_rank)
                recv_handles.put((handle, acts_tensor))
                if recv_handles.qsize()>4:
                    handle, tensor = recv_handles.get()
                    handle.wait()
                    self.acts_queue.put(tensor.to(self.device))
        while not recv_handles.empty():
            handle, tensor = recv_handles.get()
            handle.wait()
            self.acts_queue.put(tensor.to(self.device))
        del acts_tensor
    
    def grads_receiver(self):
        chunks = len(self.batches)
        dtype = torch.float16 if self.fp16 else torch.float32
        recv_handles = Queue()

        for task,index in self.schedule:
            if task == 2:
                bwd_grad_shape = self.bwd_grad_shape
                if index == (chunks-1) and self.last_chunk_size > 0:
                    bwd_grad_shape = list(self.bwd_grad_shape)
                    bwd_grad_shape[0] = self.last_chunk_size
                grads_tensor = torch.ones(bwd_grad_shape, dtype=dtype)
                handle = dist.irecv(grads_tensor, src=self.send_rank)
                recv_handles.put((handle, grads_tensor))
                if recv_handles.qsize()>4:
                    handle, tensor = recv_handles.get()
                    handle.wait()
                    self.grads_queue.put(tensor.to(self.device))
        while not recv_handles.empty():
            handle, tensor = recv_handles.get()
            handle.wait()
            self.grads_queue.put(tensor.to(self.device))
        del grads_tensor

    def acts_sender(self):
        count = 0
        for task,index in self.schedule:
            if task == 0:
                count += 1
        # count = len(self.batches)   # worsens performance if used instead of for loop. Why?

        send_handles = Queue()

        # wait_handler = Thread(target=self.handle_wait, args=(send_handles, count))
        # wait_handler.daemon=True
        # wait_handler.start()
        
        while count > 0:
            output_acts = self.acts_send_queue.get()
            handle = dist.isend(output_acts.cpu(), dst=self.send_rank)
            send_handles.put(handle)
            if send_handles.qsize()>4:
                handle = send_handles.get()
                handle.wait()
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()
        # wait_handler.join()

    def grads_sender(self):
        count = 0
        for task,index in self.schedule:
            if task == 2:
                count += 1
        # count = len(self.batches)   # worsens performance if used instead of for loop. Why?
        
        send_handles = Queue()

        # wait_handler = Thread(target=self.handle_wait, args=(send_handles, count))
        # wait_handler.daemon=True
        # wait_handler.start()

        while count > 0:
            input_grads = self.grads_send_queue.get()
            handle = dist.isend(input_grads.cpu(), dst=self.receive_rank)
            send_handles.put(handle)
            if send_handles.qsize()>4:
                handle = send_handles.get()
                handle.wait()
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()
        # wait_handler.join()
        
    # tells the model where to send acts and gradients
    def set_model_send_fn(self, recompute = False):
        def send(tensor, grads = False):
            if grads:
                self.grads_send_queue.put(tensor)
            else:
                if not recompute:
                    self.acts_send_queue.put(tensor)
        
        self.partitioned_model.set_send_fn(send)

    # tells the model how to receive acts and gradients
    def set_model_recv_fn(self, recompute = False):
        if recompute:
            ctx, acts = self.recompute_queue.get()
            if self.stage > 0:
                acts = acts.to(self.device)
            restore_rng_states(ctx, self.device)

        else:
            recv_time_start = time.time()
            acts = self.acts_queue.get() if self.stage > 0 else None
            acts = acts.to(self.device) if self.stage > 0 else None
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                recv_time = time.time() - recv_time_start
                self.logfile.write("{} {} {} {}\n".format("recvacts", 0, recv_time_start, recv_time))

        def recv(grads = False):
            if grads:
                recv_time_start = time.time()
                g = self.grads_queue.get()
                if self.make_logfile:  
                    torch.cuda.synchronize(self.device)
                    recv_time = time.time() - recv_time_start 
                    self.logfile.write("{} {} {} {}\n".format("recvgrads", 0, recv_time_start, recv_time))
                self.back_start_times.put(time.time())
                return g
            else:
                return acts
        
        self.partitioned_model.set_recv_fn(recv)
        # because there's no peek/front method for these queues
        return acts

    def worker(self, task, grad_mode, inputs_as_dict):
        """ Main body of worker loop """
        if task == 0:       
            torch.set_grad_enabled(grad_mode)

            rng_states=None
            if grad_mode == False:
                # if these acts are going to be recomputed
                rng_states = save_rng_states(self.device)

            self.set_model_send_fn(recompute = False)
            acts = self.set_model_recv_fn(recompute = False)
            task_time_start = time.time()
            output = self.model(**inputs_as_dict)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                task_time = time.time() - task_time_start
                self.logfile.write("{} {} {} {}\n".format(TASK[0], 0, str(task_time_start), str(task_time)))

            if grad_mode == False:
                if self.stage > 0:
                    acts = acts.cpu()
                ctx = (rng_states, acts)
                self.recompute_queue.put(ctx)
            else:
                # save loss and input activations for the backward pass to use
                self.loss = output[0] if isinstance(output,tuple) else output

        elif task == 1:
            torch.set_grad_enabled(True)
            self.set_model_send_fn(recompute = True)
            self.set_model_recv_fn(recompute = True)
            task_time_start = time.time()
            output = self.model(**inputs_as_dict)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                task_time = time.time() - task_time_start
                self.logfile.write("{} {} {} {}\n".format(TASK[1], 0, str(task_time_start), str(task_time)))

            self.loss = output[0] if isinstance(output,tuple) else output
        
        else:
            if self.stage != self.partitions-1:
                grads = torch.ones(self.loss.size(), dtype = torch.float32).to(self.device)
                if self.fp16:
                    # task_time_start = time.time()
                    with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True, last_partition=False) as scaled_loss:
                        scaled_loss.backward(grads)
                    if self.make_logfile:
                        torch.cuda.synchronize(self.device)
                        task_time_start = self.back_start_times.get()
                        task_time = time.time() - task_time_start
                        self.logfile.write("{} {} {} {}\n".format(TASK[2], 0, str(task_time_start), str(task_time)))
                else:
                    self.loss.backward(grads)

            else:
                chunks = len(self.batches)
                # self.loss = self.loss/chunks
                self.average_loss += (self.loss.item()/chunks)

                if self.fp16:
                    task_time_start = time.time()
                    with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True) as scaled_loss:
                        scaled_loss.backward()
                    if self.make_logfile:
                        torch.cuda.synchronize(self.device)
                        task_time = time.time() - task_time_start
                        self.logfile.write("{} {} {} {}\n".format(TASK[2], 0, str(task_time_start), str(task_time)))

                    # self.optimizer.backward(self.loss)
                else:
                    self.loss.backward()

            # print(self.stage, 'backward done')
            del self.loss
            self.loss = None
        
    def run(self):
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting pipeline')

        self.spawn_receive_workers()

        batchstart = time.time()
        '''        
        for index, task in enumerate(self.schedule):
            grad_mode = False
            if task[0] == 0:
                if self.schedule[index+1][0] == 2:      
                    # if next task in schedule is backward  -- no recomputation
                    grad_mode=True

            self.worker(task[0], grad_mode, self.batches[task[1]])

            # torch.cuda.synchronize(self.device)
            # task_time = time.time() - task_time
            
            # if self.make_logfile:
            #     self.logfile.write("{} {} {} {}\n".format(TASK[task[0]],task[1], str(task_time_start), str(task_time)))
        '''        


        # dynamic schedule - run forward if gradients for backward are not ready yet

        schedule = [s for s in enumerate(self.schedule)]
        i=0
        count_fwd = 0
        while (i<len(schedule)):
            grad_mode = False
            index, task = schedule[i]
            if (task[0]==1 and count_fwd<len(self.batches) and self.grads_queue.empty()):
            # if (task[0]==1 and count_fwd<len(self.batches) and not self.acts_queue.empty()):
                j=i
                while (j<len(schedule)):
                    if (schedule[j][1][0]==0):
                        index, task = schedule[j]
                        schedule.insert(i, schedule[j])
                        del schedule[j+1]
                        break
                    j+=1
            if (task[0]==0):
                count_fwd+=1
                if (self.schedule[index+1][0]==2):      # if next task in schedule is backward  -- no recomputation
                    grad_mode=True
            
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} task:{task[0]} {task[1]}/{len(self.batches)}')
            self.worker(task[0], grad_mode, self.batches[task[1]])

            i+=1
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} going to share embedding grads')
        
        if self.partitions > 1 and self.shared_weights is not None:
            embed_comm_start = time.time()
            share_weight_grads(self, self.tied_group)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                embed_comm_time = time.time() - embed_comm_start
                self.logfile.write("{} {} {} {}\n".format("embedcomm", 0, embed_comm_start, embed_comm_time))
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} shared embedding grads')

        # dist.barrier()

        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} crossed barrier, starting all-reduce')

        overflow = False
        global_grad_norm = -1
        if self.fp16 or self.data_parallel:
            sync_start_time = time.time()
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} all-reduce')
            overflow, global_grad_norm = self.all_reduce_opt_grads()
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                sync_time =  time.time() - sync_start_time
                self.logfile.write("all-reduce {} {} {}".format(0, sync_start_time, sync_time))
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} all-reduce done; should optimize if grads did not overflow')

        batchtime = time.time()-batchstart
        if self.make_logfile:
            self.logfile.write("\n\nBATCH END {} {}\n\n".format(batchstart, batchtime))
            self.logfile.close()        
        if self.acts_receive_thread is not None:
            self.acts_receive_thread.join()
        if self.grads_receive_thread is not None:
            self.grads_receive_thread.join()

        if self.acts_send_thread is not None:
            self.acts_send_thread.join()
        if self.grads_send_thread is not None:
            self.grads_send_thread.join()

        # return loss
        return self.average_loss, overflow, global_grad_norm

    def all_reduce_opt_grads(self):
        allred_init_start = time.time()
        # 1. allocate an uninitialized buffer for flattened gradient
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        flat_grad_size = sum(p.numel() for p in master_grads)
        flat_raw = torch.empty(flat_grad_size, device=self.device, dtype=torch.float16)
        if self.fp16:
            scaler = _amp_state.loss_scalers[0]
            loss_scale = scaler.loss_scale()
        else:
            loss_scale = 1

        chunks = len(self.batches)
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf = torch.cuda.IntTensor([0])
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (self.data_depth*chunks))
        
        if self.make_logfile:
            torch.cuda.synchronize(self.device)
            allred_init_time = time.time() - allred_init_start
            self.logfile.write("all_reduce_init {} {} {}\n".format(0, allred_init_start, allred_init_time))

        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting gradient all-reduce')

        if self.data_parallel:
            # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
            allred_time_start = time.time()
            # if self.rank == 0:
            #     print("issue allreduce 1",flush=True)
            torch.distributed.all_reduce(flat_raw, group=self.process_group)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                x = flat_raw[0] + 0
                allred_time = time.time() - allred_time_start
                self.logfile.write("all-reduce size {}\n".format(flat_grad_size))
                self.logfile.write("SYNC! all_reduce {} {} {}\n".format(flat_grad_size,allred_time_start,allred_time))
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} gradient all-reduce done')
            
        # 4. combine unscaling and unflattening of allreduced gradient
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./loss_scale)

        local_grad_norm = multi_tensor_applier(amp_C.multi_tensor_l2norm,
                                             torch.cuda.IntTensor([0]),
                                             [master_grads], False)[0]
        extra_norm_sq = 0.0
        for i,w in enumerate(self.shared_weights):
            recv_stage, send_stage = self.shared_weight_stages[i]
            if recv_stage == send_stage:
                continue
            if self.stage == send_stage:
                for p in self.parameter_names:
                    if self.parameter_names[p] == w[1]:
                        extra_norm_sq += torch.norm(p.grad) ** 2
                        break
        local_grad_norm_sq = (local_grad_norm ** 2) - extra_norm_sq
        
        # TODO: perform all-reduce for grad norm computation separately for fp32 (without overflow buf)
        

        if self.fp16:
            # 5. update loss scale
            if overflow_buf:
                print("Overflow at rank", self.rank)
            scaler = _amp_state.loss_scalers[0]
            # all-reduce to sync overflow
            if self.partitions > 1:
                osync_time_start = time.time()
                # if self.rank == 0:
                #     print("issue allreduce 2", flush=True)
                # torch.distributed.all_reduce(overflow_buf, group=self.pipeline_group)
                
                overflow_buf = overflow_buf.to(torch.float32)
                allred_tensor = torch.cat((overflow_buf, local_grad_norm_sq))
                if log_verbose:
                    print(f'{self.rank} {self.rank_within_stage} starting overflow all_reduce')
                torch.distributed.all_reduce(allred_tensor, group=self.pipeline_group)
                if log_verbose:
                    print(f'{self.rank} {self.rank_within_stage} overflow all_reduce done')
                overflow_buf = allred_tensor[0]
                global_grad_norm_sq = allred_tensor[1]
                global_grad_norm = allred_tensor[1] ** 0.5

                if self.local_rank == 0:
                    print("global norm at rank",self.rank,"is",global_grad_norm)

                if self.make_logfile:
                    x = overflow_buf.item() + 1
                    torch.cuda.synchronize(self.device)
                    osync_time = time.time() - osync_time_start
                    self.logfile.write("overflow 0 {} {}\n".format(osync_time_start, osync_time))
            else:
                global_grad_norm = local_grad_norm
                global_grad_norm_sq = local_grad_norm**2
            
            clipped = clip_grad_norm(amp.master_params(self.optimizer), global_grad_norm_sq, 1.0)
            if clipped:
                global_grad_norm = global_grad_norm/global_grad_norm # * args.clip_grad (max_norm)

            if overflow_buf.item()==0:
                overflow_buf = torch.cuda.IntTensor([0])
            else:
                overflow_buf = torch.cuda.IntTensor([1])
            
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overflow_buf = old_overflow_buf
        else:
            had_overflow = False

        # return had_overflow
        return had_overflow, global_grad_norm

def scatter(input, batch_size, chunk_size):
    """
    Accepts input dictionary and splits into microbatches
    """
    assert isinstance(input,dict) , "varuna inputs must be given as a dictionary" 
    
    microbatches = []
    num_microbatches = math.ceil(batch_size / chunk_size)
    for k,v in input.items():
        # TODO: what will happen for indivisibilities in uneven data parallelism !!
        # print(dist.get_rank(),k,v.size())
        # special case for GPT-2 attention mask
        if v.size(0) == 1:
            chunked_values = [v for _ in range(num_microbatches)]
        else:
            chunked_values = v.split(chunk_size)
        for i,value in enumerate(chunked_values):
            if len(microbatches) <= i:
                microbatches.append(dict())
            microbatches[i][k]=value
    
    return microbatches


def load_varuna_optimizer(optimizer, my_stage, num_stages, total_num_pstages, parameter_names, \
                        common_store, fp16=False, pstages_to_read = None):
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    # reload state
    opt_state = {}
    for i in pstages_to_read:
        state_ = torch.load(os.path.join(common_store,"opt-state-{}".format(i)),map_location='cpu')
        opt_state.update(state_)
    for p in amp.master_params(optimizer):
        name = parameter_names[p]
        if name in opt_state:
            optimizer.state[p] = opt_state[name]
    # reload master params
    if fp16:
        saved_master_params = dict()
        for i in pstages_to_read:
            params_ = torch.load(os.path.join(common_store, "opt-fp32-params-{}".format(i)),map_location="cpu")
            saved_master_params.update(params_)
        for p in amp.master_params(optimizer):
            name = parameter_names[p]
            if name in saved_master_params:
                p.data.copy_(saved_master_params[name].data)


def clip_grad_norm(parameters, grad_norm_sq, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    total_norm = grad_norm_sq.item() ** (1. / norm_type)
    # print(f'clip_grad_norm() total_norm = {total_norm}')
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
            
    return clip_coef<1