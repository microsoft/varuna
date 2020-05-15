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

from .partitioned_model import PartitionedModel
import gc
import numpy
# from hashlib import sha1

import os
import sys
import time

Module = nn.Module

TASK = ["fwd", "rec", "bwd"]

def share_weights(model):
    baseModule = model.model.module
    rank_within_stage = model.stage_to_rank_map[model.stage].index(model.rank)
    for i,w in enumerate(model.shared_weights):
        recv_stage, send_stage = model.shared_weight_stages[i]
        if recv_stage == send_stage:
            continue
        if model.stage == send_stage:
            recv_rank = model.stage_to_rank_map[recv_stage][rank_within_stage]
            for n,p in baseModule.named_parameters():
                if n == w[1]:
                    send_weight = p
                    break
            # print("sending", w[1])
            dist.send(send_weight.cpu(), recv_rank)
        elif model.stage == recv_stage:
            send_rank = model.stage_to_rank_map[send_stage][rank_within_stage]
            for n,p in baseModule.named_parameters():
                if n == w[0]:
                    recv_weight = p
                    break
            # print("receiving", w[0])
            recv_weight = torch.zeros(list(recv_weight.size()),dtype=torch.float16 if model.fp16 else toch.float32)
            dist.recv(recv_weight, send_rank)
            attr_names = w[0].split(".")
            param = baseModule
            for a in attr_names:
                param = getattr(param,a)
            param.data.copy_(recv_weight.data)       

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

        # partition model based on "CutPoint"s using a dry run with dummy inputs (dict)
        self.model = PartitionedModel(model, self.rank, self.local_rank, device, self.stage_to_rank_map, self.fp16, shared_weights)
        self.model.initialize( dummy_inputs, from_cache=True )
        if self.local_rank==0:
            # print("varuna init() after dry run init: ", self.local_rank, torch.cuda.memory_summary(self.device))
            print('varuna init() after dry run:', self.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        self.partitioned_model = self.model
        self.shared_weight_stages = self.model.shared_weight_stages if self.shared_weights is not None else None

        # print("SHARED WEIGHTS ARE")
        # print(self.shared_weight_stages)

        # assert(batch_size % data_depth == 0, "Batch size not divisible by data parallel depth!")
        self.batch_size = batch_size // data_depth
        self.micro_batch_size = chunk_size
        self.last_chunk_size = self.batch_size % chunk_size 
        self.init_communication(rank_within_stage)

        self.model.to(self.device)
        # torch.cuda.synchronize()
        if self.local_rank==0:
            # print("varuna_init() post model.to: ", self.local_rank, torch.cuda.memory_summary(self.device))
            print('varuna init() post model.to:', self.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        self.init_distributed()
        # if self.local_rank==0:
        #     # print("varuna_init() post ddp init: ", self.local_rank, torch.cuda.memory_summary(self.device))
        #     print('varuna init() post ddp init:', self.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())


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
            "make_logfile": True, # bool(self.rank == self.stage_to_rank_map[self.stage][-1]),
            "last_chunk_size": self.last_chunk_size,
            "shared_weights": self.shared_weights,
            "shared_weight_stages": self.shared_weight_stages,
            "stage_to_rank_map": self.stage_to_rank_map,
            "local_rank": self.local_rank
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
                process_groups[stage] = dist.new_group(ranks=ranks, backend='nccl')
            else:
                process_groups[stage] = None

        if process_groups[self.stage] is not None:
            self.process_group = process_groups[self.stage]

        # get stream to rank map
        depth = len(self.stage_to_rank_map[self.stage])
        world_size = depth * self.partitions

        if depth == 1:
            self.pipeline_group = None
            return

        stream_to_rank_map = {}
        for i in range(depth):
            stream = []
            for stage in range(self.partitions):
                stream.append(self.stage_to_rank_map[stage][i])
            stream_to_rank_map[i] = stream
        
        print("stream to rank map = ", stream_to_rank_map)
        self.pipeline_group = None
        pipeline_groups = {}
        for stream in range(depth):
            ranks = stream_to_rank_map[stream]
            # ranks = [stage_to_rank_map[s][stream] for s in range(self.partitions) ]
            if len(ranks) > 1:
                pipeline_groups[stream] = dist.new_group(ranks=ranks)
            else:
                pipeline_groups[stream] = None
            
        current_stream = self.stage_to_rank_map[self.stage].index(self.rank)
        print("this rank ", self.rank, "is part of pipeline stream ", current_stream)
        if pipeline_groups[current_stream] is not None:
            self.pipeline_group = pipeline_groups[current_stream]

    def forward(self, inputs):
        if self.fp16:
            assert self.optimizer is not None, "For fp16, you must set the optimizer using set_optimizer()"        
        
        # Divide a mini-batch into micro-batches.
        batches = scatter(inputs, int(self.batch_size),self.micro_batch_size)
        
        # need not pass the first argument if rank!=0
        # avoid dataloader compute in machines other than the first
        # ask the model writer to pass the input batch generating dataloader function to Varuna::__init__
        # and Varuna can take care of input dataloader explicitly
        self.config["make_logfile"] = bool(self.config["make_logfile"] and self.step < 730)
        pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer)
        loss, overflow = pipeline.run()
        self.step += 1
        return loss, overflow

    def evaluate(self, inputs):
        assert isinstance(inputs, dict), "input must be a dictionary!"

        share_weights(self)

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

    def checkpoint_optimizer(self, optimizer, parameter_to_name, param_name_to_pstage, cp_dir_name, tempdir=None):
        cp_time = time.time()

        # one worker from each partition
        if self.rank == self.stage_to_rank_map[self.stage][0]:
            cuts_per_stage = self.partitioned_model.cuts_per_stage
            # save param states for each cutpoint separately
            pstages = range(cuts_per_stage * self.stage, (self.stage+1)* cuts_per_stage)
            pstage_state_dicts = dict()
            for i in pstages:
                pstage_state_dicts[i] = dict()

            # store state by param names instead of actual parameters
            for key in optimizer.state:
                param_name = parameter_to_name[key]
                assert param_name in param_name_to_pstage, "param {} not found in rank {}".format(param_name,dist.get_rank())
                pstage = param_name_to_pstage[param_name]
                pstage_state_dicts[pstage][param_name] = optimizer.state[key]
            
            if tempdir is not None:
                for i in pstages:
                    temp_name =  os.path.join(tempdir,"opt-state-" + str(i))
                    torch.save(pstage_state_dicts[i], temp_name)
                    os.system("mv {} {} &".format(temp_name, cp_dir_name))
            else:
                for i in pstages:
                    torch.save(pstage_state_dicts[i], os.path.join(cp_dir_name,"opt-state-" + str(i)))

            # also store optimizer master params for mixed precision training
            if self.fp16:

                pstage_state_dicts = dict()
                for i in pstages:
                    pstage_state_dicts[i] = dict()

                for p in amp.master_params(optimizer):
                    param_name = parameter_to_name[p]
                    # not a part of the worker's stage
                    if param_name not in param_name_to_pstage:
                        continue
                    pstage = param_name_to_pstage[param_name]
                    if pstage not in pstages:
                        continue
                    pstage_state_dicts[pstage][param_name] = p
            
                if tempdir is not None:
                    for i in pstages:
                        temp_name =  os.path.join(tempdir,"opt-fp32-params-" + str(i))
                        torch.save(pstage_state_dicts[i], temp_name)
                        os.system("mv {} {} &".format(temp_name, cp_dir_name))
                else:
                    for i in pstages:
                        torch.save(pstage_state_dicts[i], os.path.join(cp_dir_name,"opt-fp32-params-" + str(i)))

        cp_time = time.time() - cp_time
        print("Opt ckpt time", cp_time)

    
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

        self.model = model
        self.partitioned_model = self.model#.module if self.data_parallel else self.model
        self.device = config["device"]
        self.schedule = schedule
        self.fp16 = config["fp16"]
        self.rank = dist.get_rank()

        self.fwd_inp_shape = config["fwd_inp_shape"]
        self.bwd_grad_shape = config["bwd_grad_shape"]

        self.shared_weights = config["shared_weights"]
        self.shared_weight_stages = config["shared_weight_stages"]
        self.stage_to_rank_map = config["stage_to_rank_map"]
        self.local_rank = config["local_rank"]

        self.make_logfile = config["make_logfile"]
        if self.make_logfile:
            replica_num = self.stage_to_rank_map[self.stage].index(self.rank)
            microBS = self.fwd_inp_shape[0] if self.bwd_grad_shape is None else self.bwd_grad_shape[0]
            logfilename = "varuna_logs-mBS" + str(microBS) + "-stage" + str(self.stage) + "of" + str(self.partitions) + "_" + str(replica_num)
            self.logfile = open(logfilename,"a")
            self.logfile.write("start time {}\n".format(time.time()))
        
        if self.partitions > 1 and self.shared_weights is not None:
            embed_comm_start = time.time()
            share_weights(self)
            # torch.cuda.synchronize(self.device)
            embed_comm_time = time.time() - embed_comm_start
            if self.make_logfile:
                self.logfile.write("{} {} {} {}\n".format("embedcomm", 0, embed_comm_start, embed_comm_time))


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
    
    # def handle_wait(self, handles, count):
    #     while count>0:
    #         handle = handles.get()
    #         handle.wait()
    #         count -= 1
    
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
            recv_time = time.time() - recv_time_start
            if self.make_logfile:
                self.logfile.write("{} {} {} {}\n".format("recvacts", 0, recv_time_start, recv_time))

        def recv(grads = False):
            if grads:
                recv_time_start = time.time()
                g = self.grads_queue.get()
                recv_time = time.time() - recv_time_start
                if self.make_logfile:   
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
            # if self.make_logfile:
            #     torch.cuda.synchronize(self.device)
            task_time = time.time() - task_time_start
            if self.make_logfile:
                self.logfile.write("{} {} {} {}\n".format(TASK[0], 0, str(task_time_start), str(task_time)))

            if grad_mode == False:
                if self.stage > 0:
                    acts = acts.cpu()
                ctx = (rng_states, acts)
                self.recompute_queue.put(ctx)
            else:
                # save loss and input activations for the backward pass to use
                self.loss = output[0] if isinstance(output,tuple) else output
            # print(self.stage, 'forward done')

            
        elif task == 1:
            torch.set_grad_enabled(True)
            self.set_model_send_fn(recompute = True)
            self.set_model_recv_fn(recompute = True)
            task_time_start = time.time()
            output = self.model(**inputs_as_dict)
            # if self.make_logfile:
            #     torch.cuda.synchronize(self.device)
            task_time = time.time() - task_time_start
            if self.make_logfile:
               self.logfile.write("{} {} {} {}\n".format(TASK[1], 0, str(task_time_start), str(task_time)))

            self.loss = output[0] if isinstance(output,tuple) else output
        
        else:
            if self.stage != self.partitions-1:
                grads = torch.ones(self.loss.size(), dtype = torch.float32).to(self.device)
                if self.fp16:
                    # task_time_start = time.time()
                    with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True, last_partition=False) as scaled_loss:
                        scaled_loss.backward(grads)
                    # if self.make_logfile:
                    #     torch.cuda.synchronize(self.device)
                    task_time_start = self.back_start_times.get()
                    task_time = time.time() - task_time_start
                    if self.make_logfile:
                        self.logfile.write("{} {} {} {}\n".format(TASK[2], 0, str(task_time_start), str(task_time)))
                else:
                    self.loss.backward(grads)

            else:
                chunks = len(self.batches)
                self.loss = self.loss/chunks
                self.average_loss += self.loss.item()

                if self.fp16:
                    task_time_start = time.time()
                    with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True) as scaled_loss:
                        scaled_loss.backward()
                    # if self.make_logfile:
                    #     torch.cuda.synchronize(self.device)
                    task_time = time.time() - task_time_start
                    if self.make_logfile:
                        self.logfile.write("{} {} {} {}\n".format(TASK[2], 0, str(task_time_start), str(task_time)))

                    # self.optimizer.backward(self.loss)
                else:
                    self.loss.backward()

            # print(self.stage, 'backward done')
            del self.loss
            self.loss = None
        
    def run(self):
        self.spawn_receive_workers()

        batchstart = time.time()
        
        for index, task in enumerate(self.schedule):
            # if self.local_rank==0  and (task[1]%200==0 or task[1]<10):
            #     print(TASK[task[0]], 'iteration memory at micro-step', task[1], ':', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
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
        

        # dynamic schedule - run forward if gradients for backward are not ready yet
        '''
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
            
            self.worker(task[0], grad_mode, self.batches[task[1]])

            i+=1
        '''


        
        if self.fp16 and self.data_parallel:
            sync_time = time.time()
            overflow = self.all_reduce_opt_grads()
            # torch.cuda.synchronize(self.device)
            sync_time =  time.time() - sync_time
            if self.make_logfile:
                self.logfile.write("SYNC! all-reduce time {}".format(sync_time))

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
        return self.average_loss, overflow

    def all_reduce_opt_grads(self):
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        # print("all_reduce_opt_grads() grad details: ", self.stage, len(master_grads), numpy.array([numpy.array(x.size()).prod() for x in master_grads]).sum())
        flat_grad_size = sum(p.numel() for p in master_grads)
        flat_raw = torch.empty(flat_grad_size, device=self.device, dtype=torch.float32)
        
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf = torch.cuda.IntTensor([0]) # not checking for overflow manually
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / self.data_depth)
        
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        allred_time_start = time.time()
        torch.distributed.all_reduce(flat_raw, group=self.process_group)
        allred_time = time.time() - allred_time_start
        if self.make_logfile:
            self.logfile.write("SYNC! all_reduce {} {} {}\n".format(flat_grad_size,allred_time_start,allred_time))
        
        # 4. combine unscaling and unflattening of allreduced gradient
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./scaler.loss_scale())
            
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]

        # all-reduce to sync overflow
        osync_time_start = time.time()
        torch.distributed.all_reduce(overflow_buf, group=self.pipeline_group)
        osync_time = time.time() - osync_time_start
        if self.make_logfile:
            self.logfile.write("overflow sync {} {}\n".format(osync_time_start,osync_time))
        if overflow_buf.item()==0:
            overflow_buf = torch.cuda.IntTensor([0])
        else:
            overflow_buf = torch.cuda.IntTensor([1])
        
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overflow_buf = old_overflow_buf

        return had_overflow

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


def load_varuna_optimizer(optimizer, my_stage, num_stages, total_num_pstages, parameter_names, common_store, fp16=False):
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
