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
from . import utils
# from .checkpoint import write_varuna_checkpoint, get_local_ckpt_tracker, \
#          load_varuna_checkpoint, load_varuna_optimizer, num_params_written, get_prev_checkpoint
import gc
import numpy
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
    Wrapper class for Varuna training.
    Args:
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

        self.rank = dist.get_rank()
        self.local_rank = local_rank if local_rank != -1 else self.rank
        self.stage_to_rank_map = utils.parse_stage_to_rank_map(stage_to_rank_map)
        self.partitions, data_depth = utils.get_varuna_config(stage_to_rank_map)
        self.stage, rank_within_stage = utils.get_this_rank_config_varuna(stage_to_rank_map, self.rank)
        self.manager_ip, self.manager_port = utils.get_heartbeat_server_info()
        self.rank_within_stage = rank_within_stage
        if self.stage == -1:
            raise ValueError("Rank " + str(self.rank) + " not found in stage to rank map!")
        self.data_parallel = data_depth > 1

        model_in_cpu = not next(model.parameters()).is_cuda
        assert model_in_cpu, "Model should be on CPU before passing to varuna!"
        assert isinstance(dummy_inputs, dict), "Sample inputs should be a dictionary!"
        for key in dummy_inputs:
            val = dummy_inputs[key]
            if isinstance(val, torch.Tensor) and val.is_cuda:
                dummy_inputs[key] = val.cpu()

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
        self.model.initialize( dummy_inputs, from_cache=True )
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
        # self.configure_checkpointing(dummy_inputs)

        self.config = {
            "stage": self.stage,
            "partitions": self.partitions,
            "fp16": self.fp16,
            "fwd_inp_shape": self.fwd_inp_shape,
            "bwd_grad_shape": self.bwd_grad_shape,
            "receive_rank": self.receive_rank,
            "send_rank": self.send_rank,
            "device": self.device,
            "data_depth": data_depth,
            "dp_process_group": self.dp_group, 
            "pipeline_process_group": self.pipeline_group,
            "tied_group": self.tied_group,
            "make_logfile": False,
            "last_chunk_size": self.last_chunk_size,
            "shared_weights": self.shared_weights,
            "shared_weight_stages": self.shared_weight_stages,
            "stage_to_rank_map": self.stage_to_rank_map,
            "local_rank": self.local_rank,
            "chunk_size": chunk_size,
            "rank_within_stage": rank_within_stage,
            "dummy_attention_mask": dummy_inputs["attention_mask"].cpu()
        }

        chunks = math.ceil(self.batch_size / self.micro_batch_size)
        self.schedule = utils.generate_schedule(chunks, self.stage, self.partitions)
        self.iteration = 0
        self.current_step = 0

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
        if self.stage < (self.partitions-1):
            self.bwd_grad_shape = self.model.backward_grad_shapes[0]
            self.bwd_grad_shape[0] = self.micro_batch_size

    def init_distributed(self):
        # create same process groups on all ranks
        
        # data parallel groups
        self.dp_group = None
        dp_groups = {}
        for stage in range(self.partitions):
            ranks = self.stage_to_rank_map[stage]
            if len(ranks) > 1:
                dp_groups[stage] = dist.new_group(ranks=ranks,backend='nccl')
            else:
                dp_groups[stage] = None
        if dp_groups[self.stage] is not None:
            self.dp_group = dp_groups[self.stage]

        # pipeline parallel groups
        data_depth = len(self.stage_to_rank_map[self.stage])
        self.tied_group = None
        self.pipeline_group = None
        pipeline_groups = {}
        tied_groups = {}
        for replica in range(data_depth):
            ranks = [self.stage_to_rank_map[i][replica] for i in range(self.partitions)]
            if len(ranks) > 1:
                pipeline_groups[replica] = dist.new_group(ranks=ranks)
                recv_stage, send_stage = self.shared_weight_stages[0]
                tied_ranks = [ranks[recv_stage], ranks[send_stage]]
                tied_groups[replica] = dist.new_group(ranks=tied_ranks)
            else:
                pipeline_groups[replica] = None
                tied_groups[replica] = None
            
        current_replica = self.stage_to_rank_map[self.stage].index(self.rank)
        print("this rank ", self.rank, "is part of pipeline replica ", current_replica)
        if pipeline_groups[current_replica] is not None:
            self.pipeline_group = pipeline_groups[current_replica]
            self.tied_group = tied_groups[current_replica]

    def configure_checkpointing(self, dummy_inputs):
        self.param_name_to_pstage = self.partitioned_model.parameter_names_to_cuts(dummy_inputs)
        # make temp dir for local ckpt trackers
        if self.local_rank == 0 and not os.path.exists(utils.VARUNA_TEMP_FOLDER):
            os.makedirs(utils.VARUNA_TEMP_FOLDER)

    def forward(self, inputs):
        raise RuntimeError("Varuna uses the 'step' function for both fwd/bwd together,\
                             or the 'evaluate' function for evaluation.")

    def step(self, inputs):
        assert isinstance(inputs, dict), "Varuna inputs should be a dictionary!"

        if self.fp16:
            assert self.optimizer is not None, "For fp16, you must set the optimizer using set_optimizer()"        
        
        # Divide a mini-batch into micro-batches.
        batches = utils.scatter(inputs, int(self.batch_size),self.micro_batch_size)
        
        self.config["make_logfile"] = bool(self.config["make_logfile"] and self.current_step < 5)
        batch_time = time.time()
        self.pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer)
        loss, overflow, global_grad_norm, fwd_time = self.pipeline.run()
        batch_time = time.time() - batch_time
        self.iteration += 1
        self.current_step += 1
        
        if self.current_step <= 5:
            message = "slowcheck {} {} {} {}".\
                        format(self.current_step, self.stage, self.rank_within_stage,fwd_time)
            utils.heartbeat(message, self.manager_ip, self.manager_port)
            
                    
        if self.rank == 0 and self.iteration%5==0:
            message = "progress {} {}".format(batch_time, self.iteration)
            utils.heartbeat(message, self.manager_ip, self.manager_port)

        return loss, overflow, global_grad_norm

    def get_status(self):
        return self.pipeline.status

    def evaluate(self, inputs):
        assert isinstance(inputs, dict), "input must be a dictionary!"

        dummy_attention_mask = self.config["dummy_attention_mask"]
        attention_mask = inputs["attention_mask"].cpu() if self.stage in [0,self.partitions-1] \
                            else torch.ones_like(dummy_attention_mask)
        if self.stage > 0 and self.stage < self.partitions - 1:
            dist.recv(attention_mask, src = self.receive_rank)
        if self.stage < self.partitions - 2:
            dist.send(attention_mask, dst = self.send_rank)
        attention_mask = attention_mask.to(self.device)

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
                mb["attention_mask"] = attention_mask
                output = self.partitioned_model(**mb)
                avg_output = output if avg_output is None else avg_output + output
            mb = batches[-1]
            mb["attention_mask"] = attention_mask
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

    def set_optimizer(self, optimizer, amp_opt_level="O2", loss_scale = "dynamic",
                            init_loss_scale = 2**20, min_loss_scale=None):
        self.optimizer = optimizer
        
        basemodel = self.partitioned_model.module
        parameter_names_ = dict()
        for n,p in basemodel.named_parameters():
            parameter_names_[p] = n

        if self.fp16:
            assert  loss_scale == 'dynamic' or type(loss) == float, \
                    "Loss scale must either be a floating point or the string 'dynamic'"
            
            basemodel, optimizer = amp.initialize(  basemodel, self.optimizer, opt_level=amp_opt_level, 
                                                    loss_scale=loss_scale, min_loss_scale=min_loss_scale )
            if loss_scale == 'dynamic':
                amp._amp_state.loss_scalers[0]._loss_scale = init_loss_scale
            
            self.partitioned_model.module = basemodel
            self.optimizer = optimizer

            # fp32 param names for checkpointing
            optimizer._amp_lazy_init()

            fp16_model_params = optimizer._amp_stash.all_fp16_params
            fp32_master_params = optimizer._amp_stash.all_fp32_from_fp16_params
            # print("stash lens",len(fp16_model_params), len(fp32_master_params))
            
            count = 0
            parameter_names = dict()
            for p_model, p_master in zip(fp16_model_params, fp32_master_params):
                if p_model in parameter_names_:
                    parameter_names[p_master] = parameter_names_.pop(p_model)
                    count += p_master.numel()
            # print(count, "params found in rank", self.rank)

            self.parameter_names = parameter_names
        else:
            self.parameter_names = parameter_names_

        self.config["parameter_names"] = self.parameter_names


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
    

class Pipeline:
    """ Pipeline parallelism for Varuna """

    def __init__(self, batches, model, config, schedule, optimizer):
        self.status = "init"
        self.batches = batches
        self.partitions = config["partitions"]
        self.stage = config["stage"]
        self.data_depth = config["data_depth"]
        self.data_parallel = bool(self.data_depth > 1)
        self.process_group = config["dp_process_group"]
        self.pipeline_group = config["pipeline_process_group"]
        self.tied_group = config["tied_group"]
        self.rank_within_stage = config["rank_within_stage"]
        self.chunks = len(self.batches)

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
            logfilename = "/home/rahul/gpt2-blob/gantts/vlogs-"+str(self.data_depth)+"dp-" + str(microBS) + "mBS-stage" + str(self.stage) + "of" + str(self.partitions) + "_" + str(replica_num)
            # logfilename = os.path.join("/home/varuna/gpt2-blob/perf_analysis_1.5b","stats",logfilename)
            self.logfile = open(logfilename,"a")
            self.logfile.write("start time {}\n".format(time.time()))

        self.receive_rank = config["receive_rank"]
        self.send_rank = config["send_rank"]

        self.last_chunk_size = config["last_chunk_size"]

        self.optimizer = optimizer
        self.fp16 = config["fp16"]

        # self.mem_previous = self.max_mem_previous = 0
        self.dummy_attention_mask = config["dummy_attention_mask"]
        self.attention_mask = batches[0]["attention_mask"].cpu() if self.stage in [0,self.partitions-1] \
                            else torch.ones_like(self.dummy_attention_mask)
        if self.stage > 0 and self.stage < self.partitions - 1:
            dist.recv(self.attention_mask, src = self.receive_rank)
        if self.stage < self.partitions - 2:
            dist.send(self.attention_mask, dst = self.send_rank)
        self.attention_mask = self.attention_mask.to(self.device)
        # print("attn mask", self.attention_mask)

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

        self.pre_fwd_events = []
        self.post_fwd_events = []
        self.avg_fwd_time = 0

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
                    self.acts_queue.put(tensor)
        while not recv_handles.empty():
            handle, tensor = recv_handles.get()
            handle.wait()
            self.acts_queue.put(tensor)
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
                    self.grads_queue.put(tensor)
        while not recv_handles.empty():
            handle, tensor = recv_handles.get()
            handle.wait()
            self.grads_queue.put(tensor)
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
            handle = dist.isend(output_acts, dst=self.send_rank)
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
            handle = dist.isend(input_grads, dst=self.receive_rank)
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
                self.grads_send_queue.put(tensor.cpu())
            else:
                if not recompute:
                    self.acts_send_queue.put(tensor.cpu())
        
        self.partitioned_model.set_send_fn(send)

    # tells the model how to receive acts and gradients
    def set_model_recv_fn(self, recompute = False):
        if recompute:
            ctx, acts = self.recompute_queue.get()
            if self.stage > 0:
                acts = acts.to(self.device)
            utils.restore_rng_states(ctx, self.device)

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
                g = self.grads_queue.get().to(self.device)
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
                rng_states = utils.save_rng_states(self.device)

            self.set_model_send_fn(recompute = False)
            acts = self.set_model_recv_fn(recompute = False)
            task_time_start = time.time()
            pre_fwd = torch.cuda.Event(enable_timing=True)
            post_fwd = torch.cuda.Event(enable_timing=True)
            pre_fwd.record()
            inputs_as_dict["attention_mask"] = self.attention_mask
            output = self.model(**inputs_as_dict)
            post_fwd.record()
            self.pre_fwd_events.append(pre_fwd)
            self.post_fwd_events.append(post_fwd)
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
            inputs_as_dict["attention_mask"] = self.attention_mask
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
        # dist.barrier()
        # time.sleep(10)
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting pipeline')        
        self.spawn_receive_workers()

        batchstart = time.time()

        # dynamic schedule - run forward if gradients for backward are not ready yet
        schedule = [s for s in enumerate(self.schedule)]
        i=0
        count_fwd = 0
        self.status = "pipeline"
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
        
        torch.cuda.synchronize(self.device)
        if len(self.pre_fwd_events) > 0:
            avg_fwd_time = 0.0
            for i in range(len(self.pre_fwd_events)):
                start = self.pre_fwd_events[i]
                end = self.post_fwd_events[i]
                avg_fwd_time += start.elapsed_time(end)
            avg_fwd_time = avg_fwd_time / len(self.pre_fwd_events)
            self.avg_fwd_time = avg_fwd_time

        dist.barrier()

        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} going to share embedding grads')
        
        if self.partitions > 1 and self.shared_weights is not None:
            self.status = "share_weights"
            embed_comm_start = time.time()
            share_weight_grads(self, self.tied_group)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                embed_comm_time = time.time() - embed_comm_start
                self.logfile.write("{} {} {} {}\n".format("embedcomm", 0, embed_comm_start, embed_comm_time))
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} shared embedding grads')

        overflow = False
        global_grad_norm = -1
        if self.fp16 or self.data_parallel:
            self.status = "all_reduce"
            sync_start_time = time.time()
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} all-reduce')
            overflow, global_grad_norm = self.all_reduce_opt_grads()
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                sync_time =  time.time() - sync_start_time
                self.logfile.write("all-reduce {} {} {}\n".format(0, sync_start_time, sync_time))
        
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

        self.status = "done"

        # return loss
        return self.average_loss, overflow, global_grad_norm, self.avg_fwd_time

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
            
            clipped = utils.clip_grad_norm(amp.master_params(self.optimizer), global_grad_norm_sq, 1.0)
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


def load_varuna_optimizer(optimizer, my_stage, num_stages, total_num_pstages, parameter_names, \
                        common_store, fp16=False, pstages_to_read = None):
    if pstages_to_read is None:
        stages_per_worker = total_num_pstages // num_stages
        pstages_to_read = range(stages_per_worker * my_stage, stages_per_worker * (my_stage + 1) )
    # reload state
    opt_state = {}
    for i in pstages_to_read:
        filename = os.path.join(common_store,f"opt-state-{i}")
        if os.path.exists(filename):
            state_ = torch.load(filename,map_location='cpu')
            opt_state.update(state_)
        else:
            shards = [os.path.join(common_store,f) for f in os.listdir(common_store) \
                        if f.startswith(f"opt-state-{i}_")]
            for filename in shards:
                state_ = torch.load(filename,map_location='cpu')
                opt_state.update(state_)
            
    for p in amp.master_params(optimizer):
        name = parameter_names[p]
        if name in opt_state:
            optimizer.state[p] = opt_state[name]
    # reload master params
    if fp16:
        saved_master_params = dict()
        for i in pstages_to_read:
            filename = os.path.join(common_store, "opt-fp32-params-{}".format(i))
            if os.path.exists(filename):
                params_ = torch.load(filename,map_location="cpu")
                saved_master_params.update(params_)
            else:
                shards = [os.path.join(common_store,f) for f in os.listdir(common_store) \
                            if f.startswith(f"opt-fp32-params-{i}_")]
                for filename in shards:
                    params_ = torch.load(filename,map_location="cpu")
                    saved_master_params.update(params_)
        for p in amp.master_params(optimizer):
            name = parameter_names[p]
            if name in saved_master_params:
                p.data.copy_(saved_master_params[name].data)
