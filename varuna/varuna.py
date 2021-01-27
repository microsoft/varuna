from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.multiprocessing import Process
from queue import Queue
from threading import Thread
import math
from apex import amp
from apex.amp import _amp_state
import amp_C, apex_C
from apex.multi_tensor_apply import multi_tensor_applier
import concurrent.futures
import shutil

from .partitioned_model import PartitionedModel
from . import utils
from .checkpoint import write_varuna_checkpoint, get_local_ckpt_tracker, \
         load_varuna_checkpoint, load_varuna_optimizer, num_params_written, get_prev_checkpoint
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
        self.configure_checkpointing(dummy_inputs)

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
            "rank_within_stage": rank_within_stage
        }

        chunks = math.ceil(self.batch_size / self.micro_batch_size)
        self.schedule = utils.generate_schedule(chunks, self.stage, self.partitions)
        self.iteration = 0

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
        
        self.config["make_logfile"] = bool(self.config["make_logfile"] and self.iteration < 10)
        batch_time = time.time()
        pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer)
        loss, overflow, global_grad_norm = pipeline.run()
        batch_time = time.time() - batch_time
        self.iteration += 1
            
        if self.rank == 0 and self.iteration%10==0:
            utils.heartbeat(self.iteration, self.manager_ip, self.manager_port)
        
        return loss, overflow

    def get_loss_scale(self):
        if not self.fp16:
            return None
        scaler = _amp_state.loss_scalers[0]
        loss_scale = scaler.loss_scale()
        return loss_scale    

    def evaluate(self, inputs, batch_size=None):
        assert isinstance(inputs, dict), "input must be a dictionary!"

        if batch_size is None:
            batch_size = self.batch_size
        fwd_inp_shape = list(self.fwd_inp_shape)
        fwd_inp_shape[0] = batch_size

        def send(x, grads=False):
            # print("sending to rank", self.send_rank, x.size())
            dist.send(x.cpu(), self.send_rank)
        def recv(grads=False):
            x = torch.zeros(fwd_inp_shape, dtype=torch.float16 if self.fp16 else torch.float32)
            # print("receiving from rank", self.receive_rank, x_shape)
            dist.recv(x, self.receive_rank)
            return x.to(self.device)
        self.partitioned_model.set_send_fn(send)
        self.partitioned_model.set_recv_fn(recv)
        
        with torch.no_grad():
            output = self.partitioned_model(inputs)
            
        return output

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
        if self.fp16:
            for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                param.grad = None

    """ Writes a varuna checkpoint with model parameters, optimizer state etc. 
        Each checkpoint is a directory, written under the given path.
        
        Args:
        global_store: string, path to a folder accessible by all nodes/ranks in the training job. 
                For example, path to a mounted blob storage. This is where the varuna checkpoint folder is written.
        step: int, iteration number for checkpoint. If None, it'll be taken from varuna's tracked progress.
        tempdir: string, path to a local directory to which to write checkpoints temporarily, and sync
                with the global store in the background. Lowers checkpoint write time in the critical path.
        shard: bool, whether to shard checkpoint writes over data parallel workers as well. Speeds up checkpoint 
    """
    def checkpoint(self, global_store, step=None, tempdir=None, shard=False, on_demand = False):
        if step is None:
            step = self.iteration

        ckpt_future = write_varuna_checkpoint(self, global_store, step, 
                                tempdir=tempdir, shard=shard)
        
        return ckpt_future
    
    def to(self, device):
        self.model.to(device)

    def load_checkpoint(self, global_store, iteration, check_complete = True):
        cp_dir_name = os.path.join(global_store, "varuna_ckpt_{}".format(iteration))

        if check_complete:
            num_parameter_instances = len(self.param_name_to_pstage)
            params_written = num_params_written(global_store, iteration)
            if params_written < num_parameter_instances:
                prev_ckpt = get_prev_checkpoint(global_store, iteration)
                with open(get_local_ckpt_tracker(self.local_rank),"w") as f:
                    f.write(str(prev_ckpt))
                assert False, f"CKPT NOT COMPLETE!!, only {params_written}/{num_parameter_instances} params done"

        total_num_pstages = self.partitioned_model.num_cutpoints + 1

        model_state_dict = load_varuna_checkpoint(self.stage, self.partitions, 
                                                total_num_pstages,  cp_dir_name)
        # for i,w in enumerate(self.shared_weights):
        #     recv_stage, send_stage = self.shared_weight_stages[i]
        #     recv_name, send_name = w
        #     if (recv_stage == send_stage) or (self.stage not in [recv_stage, send_stage]):
        #         continue
        #     pstage = None; pname = None
        #     if self.stage == recv_stage and recv_name not in model_state_dict:
        #         pstage = self.param_name_to_pstage[send_name]
        #         name = send_name
        #     if self.stage == send_stage and send_name not in model_state_dict:
        #         # pstage = self.param_name_to_pstage[recv_name] #TODO: FIX THIS ASAP!!
        #         pstage = 0
        #         name = recv_name
        #     if pstage is not None:
        #         print("WARNING: single checkpoint found for shared params", recv_name, send_name)
        #         state_dict_ = load_varuna_checkpoint(self.stage, self.partitions, total_num_pstages,  
        #                                             cp_dir_name, pstages_to_read = [pstage])
        #         assert name in state_dict_, f"{name} not found in any checkpoint!"
        #         model_state_dict[send_name] = state_dict_[recv_name]
        #         print(f"Renamed {recv_name} with {send_name}")
        #         # self.partitioned_model.module.load_state_dict(state_dict, strict=False)
        #         print("keys",self.partitioned_model.module.state_dict().keys())
        #         # self.partitioned_model.module.lm_head_weight
        #         # for p in self.parameter_names:
        #         #     if self.parameter_names[p] == send_name:
        #         #         p.data.copy_(state_dict[send_name].data)
        #         print(f"state dict has {model_state_dict['lm_head_weight']}")

        self.partitioned_model.module.load_state_dict(model_state_dict)
        # if self.stage == 3:
        #     self.partitioned_model.module.lm_head_weight.data.copy_(model_state_dict[send_name].data)

        load_varuna_optimizer(self.optimizer, self.stage, self.partitions, 
                              total_num_pstages, self.parameter_names, 
                              cp_dir_name, device=self.device)
        # reload master params for mixed precision
        if self.fp16:
            for p in amp.master_params(self.optimizer):
                name = self.parameter_names[p]
                if name in model_state_dict:
                    # print(f"{self.stage} loading {name}\n",end="")
                    p.data.copy_(model_state_dict[name].data)
            
        # if self.stage == 0:
        #     print(f"STAGE 0 {self.partitioned_model.module.language_model.embedding.word_embeddings.weight}")
        # if self.stage == 3:
        #     print(f'STAGE 3 {self.partitioned_model.module.lm_head_weight}')

        with open(get_local_ckpt_tracker(self.local_rank),"w") as f:
            # print("writing", iteration)
            f.write(str(iteration))

        self.iteration = iteration
                

class Pipeline:
    """ Pipeline parallelism for Varuna """

    def __init__(self, batches, model, config, schedule, optimizer):
        self.batches = batches
        self.model = model
        self.partitioned_model = self.model
        self.schedule = schedule
        self.rank = dist.get_rank()
        self.opportunistic = True

        self.read_config(config)
        self.data_parallel = bool(self.data_depth > 1)

        if self.make_logfile:
            replica_num = self.stage_to_rank_map[self.stage].index(self.rank)
            microBS = config["chunk_size"]
            logfilename = "varuna_logs-"+str(self.data_depth)+"dp-" + str(microBS) + "mBS-stage" + str(self.stage) + "of" + str(self.partitions) + "_" + str(replica_num)
            # logfilename = os.path.join("/home/varuna/gpt2-blob/perf_analysis_2.5b","stats",logfilename)
            self.logfile = open(logfilename,"a")
            self.logfile.write("start time {}\n".format(time.time()))

        self.optimizer = optimizer

        self.spawn_send_workers()
        # self.spawn_receive_workers()
        self.acts_queue = Queue()
        self.grads_queue = Queue()
      
        
        self.recompute_queue = Queue()

        # self.back_start_times = Queue()

        # communication queues
        self.partitioned_model.set_queues(self.acts_send_queue, self.grads_send_queue,
                                          self.acts_queue, self.grads_queue, self.recompute_queue  )

        # stores output of recompute(/forward) pass to be used by backward()
        self.loss = None
        self.average_loss = 0

    def read_config(self, config):

        self.partitions = config["partitions"]
        self.stage = config["stage"]
        self.data_depth = config["data_depth"]
        
        self.dp_group = config["dp_process_group"]
        self.pipeline_group = config["pipeline_process_group"]
        self.tied_group = config["tied_group"]
        self.rank_within_stage = config["rank_within_stage"]

        self.device = config["device"]
        self.fp16 = config["fp16"]

        self.fwd_inp_shape = config["fwd_inp_shape"]
        self.bwd_grad_shape = config["bwd_grad_shape"]
        self.parameter_names = config["parameter_names"]

        self.shared_weights = config["shared_weights"]
        self.shared_weight_stages = config["shared_weight_stages"]
        self.stage_to_rank_map = config["stage_to_rank_map"]
        self.local_rank = config["local_rank"]

        self.make_logfile = config["make_logfile"]
        self.receive_rank = config["receive_rank"]
        self.send_rank = config["send_rank"]
        self.last_chunk_size = config["last_chunk_size"]
    
    def spawn_receive_workers(self):
        self.acts_receive_thread = None
        self.grads_receive_thread = None

        if self.stage > 0:
            self.acts_receive_thread = Thread(target=self.acts_receiver, args=())
            self.acts_receive_thread.daemon=True
            self.acts_receive_thread.start()

        if self.stage < self.partitions-1:
            self.grads_receive_thread = Thread(target=self.grads_receiver, args=())
            self.grads_receive_thread.daemon=True
            self.grads_receive_thread.start()
    
    def spawn_send_workers(self):
        self.grads_send_queue = Queue()
        self.acts_send_queue = Queue()
        self.acts_send_thread = None
        self.grads_send_thread = None

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
        recvd = 0

        for task,index in self.schedule:
            if task == 0:
                fwd_inp_shape = self.fwd_inp_shape
                if index == (chunks-1) and self.last_chunk_size > 0:
                    fwd_inp_shape = list(self.fwd_inp_shape)
                    fwd_inp_shape[0] = self.last_chunk_size
                # print(f"{self.rank} recieving acts {fwd_inp_shape} from {self.receive_rank}\n",end="")
                acts_tensor = torch.ones(fwd_inp_shape, dtype=dtype)
                handle = dist.irecv(acts_tensor, src=self.receive_rank)
                recv_handles.put((handle, acts_tensor))
                if recv_handles.qsize()>4:
                    handle, tensor = recv_handles.get()
                    handle.wait()
                    recvd += 1
                    self.acts_queue.put(tensor)
        while not recv_handles.empty():
            handle, tensor = recv_handles.get()
            handle.wait()
            recvd += 1
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
                # print(f"{self.rank} recieving grads {bwd_grad_shape} from {self.send_rank}\n",end="")
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
        send_handles = Queue()
        sent = 0
        
        while count > 0:
            output_acts = self.acts_send_queue.get()
            # print(f"{self.rank} sending acts {output_acts.size()} to {self.send_rank}\n",end="")
            handle = dist.isend(output_acts, dst=self.send_rank)
            send_handles.put(handle)
            if send_handles.qsize()>4:
                handle = send_handles.get()
                handle.wait()
                sent += 1
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()
            sent += 1

    def grads_sender(self):
        count = 0
        for task,index in self.schedule:
            if task == 2:
                count += 1
        
        send_handles = Queue()

        while count > 0:
            input_grads = self.grads_send_queue.get()
            # print(f"{self.rank} sending grads {input_grads.size()} to {self.receive_rank}\n",end="")
            handle = dist.isend(input_grads, dst=self.receive_rank)
            send_handles.put(handle)
            if send_handles.qsize()>4:
                handle = send_handles.get()
                handle.wait()
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()
        
    def close_comm_threads(self):
        if self.acts_receive_thread is not None:
            self.acts_receive_thread.join()
        if self.grads_receive_thread is not None:
            self.grads_receive_thread.join()

        if self.acts_send_thread is not None:
            self.acts_send_thread.join()
        if self.grads_send_thread is not None:
            self.grads_send_thread.join()

    def worker(self, task, grad_mode, inputs_as_dict):
        """ Main body of worker loop """
        # forward
        if task == 0:       
            torch.set_grad_enabled(grad_mode)
            output = self.model(inputs_as_dict, save_ctx=not grad_mode, handle_comm=True)

            if grad_mode == True:
                # save loss and input activations for the backward pass to use
                self.loss = output[0] if isinstance(output,tuple) else output

        # recompute
        elif task == 1:
            torch.set_grad_enabled(True)
            output = self.model(inputs_as_dict, recompute=True, handle_comm=True)
            self.loss = output[0] if isinstance(output,tuple) else output
        
        # backward
        else:
            if self.stage == self.partitions - 1:
                chunks = len(self.batches)
                # self.loss = self.loss/chunks
                self.average_loss += (self.loss.item()/chunks)

            grads = torch.ones(self.loss.size(), dtype = torch.float32).to(self.device)
            if self.fp16:
                with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True, 
                            last_partition=self.stage == self.partitions-1) as scaled_loss:
                    scaled_loss.backward(grads)
            else:
                self.loss.backward(grads)

            self.loss = None
        
    def run(self):
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting pipeline')

        self.spawn_receive_workers()
        batchstart = time.time()

        schedule = [s for s in enumerate(self.schedule)]
        i=0
        count_fwd = 0
        while (i<len(schedule)):
            grad_mode = False
            index, task = schedule[i]
            # dynamic schedule - run forward if gradients for backward are not ready yet
            if self.opportunistic and (task[0]==1 and count_fwd<len(self.batches) and self.grads_queue.empty()):
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
                print(f'{self.stage} {self.rank_within_stage} task:{task[0]} {task[1]}/{len(self.batches)}\n', end="")
            self.worker(task[0], grad_mode, self.batches[task[1]])
            
            i+=1
        
        
        if log_verbose:
            print(f'{self.stage} {self.rank_within_stage} going to share embedding grads')
        
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

        # if log_verbose:
        #     print(f'{self.rank} {self.rank_within_stage} crossed barrier, starting all-reduce')

        overflow = False
        global_grad_norm = -1
        if (self.partitions > 1) or self.data_parallel:
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

        self.close_comm_threads()

        batchtime = time.time()-batchstart
        if self.make_logfile:
            self.logfile.write("\n\nBATCH END {} {}\n\n".format(batchstart, batchtime))
            self.logfile.close()        

        return self.average_loss, overflow, global_grad_norm

    def all_reduce_opt_grads(self):
        allred_init_start = time.time()
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        flat_raw = torch.empty(flat_grad_size, device=self.device, dtype=torch.float16 if self.fp16 else torch.float32)
        
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
            loss_scale / (self.data_depth*chunks))
        
        if self.make_logfile:
            torch.cuda.synchronize(self.device)
            allred_init_time = time.time() - allred_init_start
            self.logfile.write("all_reduce_init {} {} {}\n".format(0, allred_init_start, allred_init_time))

        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting gradient all-reduce')

        if self.data_parallel:
            allred_time_start = time.time()
            torch.distributed.all_reduce(flat_raw, group=self.dp_group)
            if self.make_logfile:
                torch.cuda.synchronize(self.device)
                x = flat_raw[0] + 0
                allred_time = time.time() - allred_time_start
                self.logfile.write("all-reduce size {}\n".format(flat_grad_size))
                self.logfile.write("SYNC! all_reduce {} {} {}\n".format(flat_grad_size,allred_time_start,allred_time))
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} gradient all-reduce done')
            
        if self.fp16:
            amp_C.multi_tensor_scale(65536,
                overflow_buf,
                [allreduced_views, master_grads],
                1./loss_scale)

        overflow_buf = overflow_buf.to(torch.float32)
        overflow_buf, global_grad_norm = self.global_overflow_and_norm(master_grads, overflow_buf if self.fp16 else None)
        global_grad_norm_sq = global_grad_norm ** 2

        clipped = utils.clip_grad_norm(amp.master_params(self.optimizer), global_grad_norm_sq, 1.0)
        if clipped:
            global_grad_norm = global_grad_norm/global_grad_norm

        if self.fp16:
            overflow_buf = torch.cuda.IntTensor([0 if overflow_buf.item()==0 else 1])
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overflow_buf = old_overflow_buf

        return had_overflow, global_grad_norm

    
    def global_overflow_and_norm(self, master_grads, overflow_buf=None):
        
        local_grad_norm = multi_tensor_applier(amp_C.multi_tensor_l2norm,
                                             torch.cuda.IntTensor([0]),
                                             [master_grads], False)[0]
        
        local_grad_norm_sq = (local_grad_norm ** 2) - self.extra_grad_norm_sq()

        if self.partitions > 1:
            osync_time_start = time.time()
            allred_tensor = local_grad_norm_sq
            if overflow_buf is not None:
                allred_tensor = torch.cat((overflow_buf, allred_tensor))
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} starting overflow all_reduce')
            torch.distributed.all_reduce(allred_tensor, group=self.pipeline_group)
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} overflow all_reduce done')
            
            if overflow_buf is not None:
                overflow_buf, global_grad_norm_sq = allred_tensor
            else:
                global_grad_norm_sq = allred_tensor
            global_grad_norm = global_grad_norm_sq ** 0.5

            if self.make_logfile:
                x = overflow_buf.item() + 1
                torch.cuda.synchronize(self.device)
                osync_time = time.time() - osync_time_start
                self.logfile.write("overflow 0 {} {}\n".format(osync_time_start, osync_time))
        
        else:
            global_grad_norm_sq = local_grad_norm_sq
            global_grad_norm = global_grad_norm_sq ** 0.5

        return overflow_buf, global_grad_norm
        
    
    def extra_grad_norm_sq(self):
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
        return extra_norm_sq


