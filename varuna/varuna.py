from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.multiprocessing import Process
from queue import Queue
from threading import Thread

from .partitioned_model import PartitionedModel

import os
import sys

Module = nn.Module

class Varuna:
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
                optimizer,
                fp16 = False,
                chunks: int=1,
                local_rank=-1):
        super().__init__()

        self.chunks = chunks
        self.partitions = len(stage_to_rank_map)
        self.rank = dist.get_rank()
        self.local_rank = local_rank if local_rank != -1 else self.rank
        
        self.stage = -1
        for stage in self.stage_to_rank_map:
            i = 0
            for rank in self.stage_to_rank_map[stage]:
                if rank == self.rank:
                    rank_within_stage = i
                    self.stage = stage
                    break
        else:
            raise ValueError("Rank " + self.rank + " not found in stage to rank map!")

        # if multiple ranks in stage:
        #     # DistributedSampler(shuffle = False)
        #     nn.DistributedDataParallel()
        #     # dataloader

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.optimizer = optimizer
        self.fp16 = fp16

        if self.fp16:
            from apex import amp

        
        # partition model based on "CutPoint"s using a dry run with dummy inputs (dict)
        self.model = PartitionedModel(model, self.rank, self.local_rank, self.local_rank, stage_to_rank_map)
        self.model.initialize( dummy_inputs )

        self.micro_batch_size = int(batch_size // chunks)
        self.init_communication(rank_within_stage, stage_to_rank_map)

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

        self.config = {
            "stage": self.stage,
            "partitions": self.partitions,
            "fp16": self.fp16,
            "fwd_inp_shape": self.fwd_inp_shape,
            "bwd_grad_shape": self.bwd_grad_shape,
            "recieve_ranks": self.recieve_ranks,
            "recieve_indices": self.recieve_indices,
            "send_ranks": self.send_ranks,
            "send_indices": self.send_indices,
            "device": self.device
        }

        self.schedule = self.generate_schedule()

    def init_communication(self, rank_within_stage, stage_to_rank_map):

        per_gpu_batch_size = self.micro_batch_size / len(stage_to_rank_map[self.stage])
        self.start = rank_within_stage * per_gpu_batch_size
        self.end = self.start + per_gpu_batch_size - 1

        self.recieve_ranks = self.send_ranks = None
        self.recieve_indices = self.send_indices = None

        # send ranks
        if self.stage < (self.partitions-1):
            depth_next = stage_to_rank_map[self.stage + 1]
            per_gpu_batch_size_next = self.micro_batch_size / depth_next

            send_rank_indices = range(self.start // per_gpu_batch_size_next, (self.end // per_gpu_batch_size_next) + 1)
            self.send_indices = []
            self.send_ranks = []

            start_ = self.start
            for i in send_rank_indices:
                end_ = max(self.end+1, per_gpu_batch_size_next * (i+1))
                self.send_indices.append((start_ - self.start, end_ - self.start) )
                self.send_ranks()
                start_ = end_

        # recieve ranks
        if self.stage > 0:
            depth_prev = stage_to_rank_map[self.stage + 1]
            per_gpu_batch_size_prev = self.micro_batch_size / depth_prev

            recieve_rank_inidices = range(self.start // per_gpu_batch_size_prev, (self.end // per_gpu_batch_size_prev) + 1 )
            self.recieve_ranks = []
            self.recieve_indices = []

            start_ = self.start
            for i in recieve_rank_inidices:
                end_ = max(self.end+1, per_gpu_batch_size_prev * (i+1))
                self.recieve_indices.append((start_ - self.start, end_ - self.start) )
                self.recieve_ranks.append(stage_to_rank_map[i])
                start_ = end_

    def run_pipeline(self, inputs):
        # Divide a mini-batch into micro-batches.
        batches = scatter(inputs, self.chunks)
        
        # need not pass the first argument if rank!=0
        # avoid dataloader compute in machines other than the first
        # ask the model writer to pass the input batch generating dataloader function to Varuna::__init__
        # and Varuna can take care of input dataloader explicitly
        pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer)
        return pipeline.run()
    
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()

    def zero_grad(self):
        self.model.zero_grad()
    
    def checkpoint(self, cpname):
        self.model.checkpoint(cpname)
    
    def generate_schedule(self):
        c_schedule = os.popen(os.path.join(os.path.dirname(os.path.abspath(__file__)),'genschedule ')+str(self.partitions)+' '+str(self.chunks)+' '+str(self.rank)).read()
        schedule = list()

        steps = c_schedule.split(';')
        steps = steps[:-1]
        for step in steps:
            task = step.split(',')
            schedule.append((int(task[0]), int(task[1])))
        
        return schedule
                

def save_rng_states():
    """capture current CPU, GPU random number generator states to reuse while recomputing activations
    in order to ensure Referential Transparency
    """
    cpu_rng_state = torch.get_rng_state()

    gpu_rng_states: Optional[ByteTensor]
    gpu_rng_states = torch.cuda.get_rng_state_all() 
    return (cpu_rng_state, gpu_rng_states)

def restore_rng_states(rng_states):
    cpu_rng_state, gpu_rng_states = rng_states
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(gpu_rng_states)        # todo: verify correctness;   batchNorm, dropouts, convlayers?


def acts_sender(rank, acts_send_queue):
    while (True):
        output_acts = acts_send_queue.get()

        # for rank, (s,e) in zip(self.send_ranks, self.send_indices):
        #     handle = dist.isend(output_acts[s:e].cpu(), dst=rank)
        #     handle.wait()

        handle = dist.isend(output_acts.cpu(), dst=rank+1)
        handle.wait()

def grads_sender(rank, grads_send_queue):
    while (True):
        input_grads = grads_send_queue.get()

        # for rank, (s,e) in zip(self.recieve_ranks, self.recieve_indices):
        #     handle = dist.isend(output_acts[s:e].cpu(), dst=rank)
        #     handle.wait()

        handle = dist.isend(input_grads.cpu(), dst=rank-1)
        handle.wait()


class Pipeline:
    """ Pipeline parallelism for Varuna """

    def __init__(self, batches, model, config, schedule, optimizer):
        self.batches = batches
        self.partitions = config["partitions"]
        self.stage = config["stage"]

        self.model = model
        self.device = config["device"]
        self.world_size = self.partitions
        self.schedule = schedule
        self.fwd_inp_shape = config["fwd_input_shape"]
        self.bwd_grad_shape = config["bwd_grad_shape"]
        
        self.recieve_ranks = config["recieve_ranks"]
        self.recieve_indices = config["recieve_indices"]
        self.send_ranks = config["send_ranks"]
        self.send_indices = config["send_indices"]

        self.optimizer = optimizer
        self.fp16 = config["fp16"]

        self.grads_send_queue = Queue()
        self.acts_send_queue = Queue()

        self.spawn_send_workers()

        self.acts_queue = Queue()       # activation at the boundary, rename as input_acts
        self.grads_queue = Queue()
        self.recompute_queue = Queue()

        self.acts_recieve_thread = None
        self.grads_recieve_thread = None

        # stores output of recompute(/forward) pass to be used by backward()
        # assuming we never encounter 'rfb'/'rrb' schedule
        self.loss = None
        self.average_loss = 0

    def spawn_recieve_workers(self):
        self.acts_recieve_thread = Thread(target=self.acts_reciever, args=())
        self.acts_recieve_thread.daemon=True
        self.acts_recieve_thread.start()

        self.grads_recieve_thread = Thread(target=self.grads_reciever, args=())
        self.grads_recieve_thread.daemon=True
        self.grads_recieve_thread.start()
    
    def spawn_send_workers(self):
        self.acts_send_thread = Thread(target=acts_sender, args=(self.stage, self.acts_send_queue))
        self.acts_send_thread.daemon=True
        self.acts_send_thread.start()

        self.grads_send_thead = Thread(target=grads_sender, args=(self.stage, self.grads_send_queue))
        self.grads_send_thead.daemon=True
        self.grads_send_thead.start() 
    
    def acts_reciever(self):
        count=0
        if self.stage != 0:
            for task,index in self.schedule:
                if task == 0:
                    count += 1
            while count > 0:
                # inp_shape = self.fwd_inp_shape
                # acts_tensor = torch.ones(self.fwd_inp_shape, dtype=torch.float32)
                # start = 0; 
                # for rank, (s,e) in zip(self.recieve_ranks, self.recieve_indices):
                #     fwd_inp_shape[0] = e-s
                #     acts_tensor_i = torch.ones(fwd_inp_shape, dtype=torch.float32)
                #     handle = dist.irecv(acts_tensor_i, src=rank)
                #     handle.wait()
                #     end = start + fwd_inp_shape[0]
                #     acts_tensor[start:end] = acts_tensor_i
                #     start = end

                acts_tensor = torch.ones(self.fwd_inp_shape, dtype=torch.float32)
                req = dist.irecv(acts_tensor, src=self.rank-1)
                req.wait()
                count -= 1
                self.acts_queue.put(acts_tensor.to(self.device))
    
    def grads_reciever(self):
        world_size = self.world_size        # todo: get world_size instead of rank?
        count = 0
        if self.stage != world_size-1:
            for task,index in self.schedule:
                if task == 2:
                    count += 1
            while count > 0:
                # grad_shape = self.bwd_grad_shape
                # grads_tensor = torch.ones(grad_shape, dtype=torch.float32)
                # start = 0; 
                # for rank, (s,e) in zip(self.recieve_ranks, self.recieve_indices):
                #     grad_shape[0] = e-s
                #     grads_tensor_i = torch.ones(grad_shape, dtype=torch.float32)
                #     handle = dist.irecv(grads_tensor_i, src=rank)
                #     handle.wait()
                #     end = start + grad_shape[0]
                #     acts_tensor[start:end] = grads_tensor_i
                #     start = end

                grads_tensor = torch.ones(self.bwd_grad_shape, dtype=torch.float32)
                req = dist.irecv(grads_tensor, src=self.rank+1)
                req.wait()              
                count -= 1
                self.grads_queue.put(grads_tensor.to(self.device))

    # tells the model where to send acts and gradients
    def set_model_send_fn(self, recompute = False):
        def send(tensor, grads = False):
            if grads:
                self.grads_send_queue.put(tensor)
            else:
                if not recompute:
                    self.acts_send_queue.put(tensor)
        
        self.model.set_send_fn(send)

    # tells the model how to recieve acts and gradients
    def set_model_recv_fn(self, recompute = False):
        if self.stage > 0:
            if recompute:
                ctx, acts = self.recompute_queue.get()
                restore_rng_states(ctx)
            else:
                acts = self.acts_queue.get()
        else:
            acts = None

        def recv(grads = False):
            if grads:
                return self.grads_queue.get()
            else:
                return acts
        
        self.model.set_recv_fn(recv)
        # because there's no peek/front method for these queues
        return acts

    def worker(self, task, grad_mode, inputs_as_dict):
        """ Main body of worker loop """
        world_size = self.world_size

        if task == 0:       
            torch.set_grad_enabled(grad_mode)

            self.set_model_send_fn(recompute = False)
            acts = self.set_model_recv_fn(recompute = False)
            output = self.model(**inputs_as_dict)

            if grad_mode == False and self.rank > 0:
                # if these acts are going to be recomputed
                rng_states = save_rng_states()
                ctx = (rng_states, acts)
                self.recompute_queue.put(ctx)
            else:
                # save loss and input activations for the backward pass to use
                self.loss = output[0] if isinstance(output,tuple) else output
        
        elif task == 1:
            torch.set_grad_enabled(True)

            self.set_model_send_fn(recompute = True)
            self.set_model_recv_fn(recompute = True)
            output = self.model(**inputs_as_dict)

            self.loss = output[0] if isinstance(output,tuple) else output
        
        else:
            if self.stage != world_size-1:
                if self.fp16:
                    with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward(grads)
                else:
                    self.loss.backward(torch.ones(self.loss.size()).to(self.device))

            else:
                chunks = len(self.batches)
                self.loss = self.loss/chunks
                self.average_loss += self.loss.item()

                if self.fp16:
                    with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.loss.backward()

        
    def run(self):
        self.spawn_recieve_workers()

        for index, task in enumerate(self.schedule):
            grad_mode = False
            if task[0] == 0:
                if self.schedule[index+1][0] == 2:      
                    # if next task in schedule is backward  -- no recomputation
                    grad_mode=True

            self.worker(task[0], grad_mode, self.batches[task[1]])
            # todo: return loss at (rank-1)th device
        
        self.acts_recieve_thread.join()
        self.grads_recieve_thread.join()

        # return loss
        if self.stage == self.world_size - 1:
            return self.average_loss
        return 0

def scatter(input, chunks):
    """
    Accepts input dictionary and splits into microbatches
    """
    # assert(isinstance(inputs,dict) , "varuna inputs must be given as a dictionary")
    
    microbatches = [dict() for _ in range(chunks)]
    for k,v in input.items():
        # print(k)
        chunked_values = v.chunk(chunks)
        for i,value in enumerate(chunked_values):
            microbatches[i][k]=value
    
    return microbatches
