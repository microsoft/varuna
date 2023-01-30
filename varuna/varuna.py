from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.multiprocessing import Process
from queue import Queue
from threading import Thread
import math
try:
    from apex import amp
    from apex.amp import _amp_state
    import amp_C, apex_C
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print("No apex!")
import concurrent.futures

from .partitioned_model import PartitionedModel
from .pipeline import Pipeline
from . import utils
from .checkpoint import write_varuna_checkpoint, get_local_ckpt_tracker, \
         load_varuna_checkpoint, load_varuna_optimizer, num_params_written, get_prev_checkpoint
import gc
import numpy
import socket

import math, shutil
import os, sys
import time

Module = nn.Module

log_verbose = False

TASK = ["fwd", "rec", "bwd"]
    
class Varuna(Module):
    r"""Module to implement varuna training. The model must be wrapped in an instance 
    of ``Varuna`` before training. This should be done before optimizer creation and the 
    :attr:`model` passed should be on CPU.

    Creating a ``Varuna`` instance profiles the model briefly using :attr:`dummy_inputs`
    and partitions it according to the distributed rank and launcher arguments.
    The partitioned model is then moved to the allocated cuda device. The profiling
    information is cached and can be re-used on resuming, unless :attr:`from_cache` is False.
    The ``Varuna`` module performs mixed precision training internally if enabled through the 
    :attr:`fp16` arg, no external handling is required. 

    :param model: The model to initialize for training.
    :type model: torch.nn.Module
    :param stage_to_rank_map: Placement of pipeline stages in the distribued job, encoded as a string. 
        Passed by ``varuna.launcher`` to each worker as an argument.
    :type stage_to_rank_map: dict
    :param get_batch_fn: Function to get sample input batches of a given size, as dictionaries. 
        These are used to profile the model structure as ``model(**get_batch_fn(k, device='cpu))``.
    :type get_batch_fn: function(size: int, device: torch.device or None)
    :param batch_size: Global batch size for the distributed training job.
    :type batch_size: int
    :param chunk_size: The micro-batch size to be used for pipeline parallelism.
    :type chunk_size: int
    :param fp16: whether to enable mixed precision training.
    :type fp16: bool
    :param local_rank: The local rank as passed by ``varuna.launcher``. If not given, 
        defaults to the global rank.
    :type local_rank: int
    :param device: index of the cuda device to use. Recommended to be the same as local_rank,
        which is the default if not specified.
    :type device: int
    :param shared_weights: A list of tuples, where each each tuple is a pair of weight names (strings),
        such that the two weights are shared in the model (see weight sharing)
    :type shared_weights: list or None
    :param from_cache: Whether to use cached profiling information if available.
    :type from_cache: bool
    
    .. note::

        Optimizer initiliastion should be done after  ``Varuna`` initialisation, so that the ``param_group`` s
        for the optimizer only contain parameters from the partitioned model. This is important both for memory 
        usage and correctness of fp16 training. Once ``Varuna`` and the optimizer are initialised, :func:`set_optimizer`
        should be called to connect the two.

    """
    def __init__(self,
                model,
                stage_to_rank_map,
                get_batch_fn,
                batch_size,
                chunk_size,
                fp16 = False, 
                local_rank=-1,
                device=-1,
                shared_weights=None,
                from_cache=True):
        super().__init__()

        self.rank = dist.get_rank()
        self.local_rank = local_rank if local_rank != -1 else self.rank
        self.stage_to_rank_map = utils.parse_stage_to_rank_map(stage_to_rank_map)
        self.partitions, self.data_depth = utils.get_varuna_config(stage_to_rank_map)
        self.stage, self.rank_within_stage = utils.get_this_rank_config_varuna(stage_to_rank_map, self.rank)
        self.manager_ip, self.manager_port = utils.get_heartbeat_server_info()

        if self.stage == -1:
            raise ValueError("Rank " + str(self.rank) + " not found in stage to rank map!")
        self.data_parallel = self.data_depth > 1

        model_in_cpu = not next(model.parameters()).is_cuda
        assert model_in_cpu, "Model should be on CPU before passing to varuna!"
        # assert isinstance(dummy_inputs, dict), "Sample inputs should be a dictionary!"
        # for key in dummy_inputs:
        #     val = dummy_inputs[key]
        #     if isinstance(val, torch.Tensor) and val.is_cuda:
        #         dummy_inputs[key] = val.cpu()

        if device == -1:
            device = self.local_rank
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            torch.cuda.set_device(device)
            self.device = torch.device("cuda", device)

        self.optimizer = None
        self.fp16 = fp16
        self.shared_weights = shared_weights

        # partition model based on "CutPoint"s using a dry run with dummy inputs (dict)
        self.model = PartitionedModel(model, self.rank, self.local_rank, device, self.stage_to_rank_map, self.fp16, shared_weights)
        self.model.initialize( get_batch_fn, from_cache=from_cache )
        self.partitioned_model = self.model
        self.shared_weight_stages = self.model.shared_weight_stages if self.shared_weights is not None else None

        print("SHARED WEIGHTS ARE")
        print(self.shared_weight_stages)

        self.batch_size = batch_size // self.data_depth
        self.micro_batch_size = chunk_size
        self.last_chunk_size = self.batch_size % chunk_size 
        self.init_communication()

        self.model.to(self.device)        
        self.init_distributed()
        self.configure_checkpointing()

        self.config = {
            "stage": self.stage,
            "partitions": self.partitions,
            "fp16": self.fp16,
            "fwd_inp_shape": self.fwd_inp_shape,
            "fwd_inp_shape_changes": self.fwd_inp_shape_changes,
            "bwd_grad_shape": self.bwd_grad_shape,
            "bwd_grad_shape_changes": self.bwd_grad_shape_changes,
            "receive_rank": self.receive_rank,
            "send_rank": self.send_rank,
            "device": self.device,
            "data_depth": self.data_depth,
            "pipeline_process_group": self.pipeline_group,
            "make_logfile": False,
            "last_chunk_size": self.last_chunk_size,
            "stage_to_rank_map": self.stage_to_rank_map,
            "local_rank": self.local_rank,
            "chunk_size": chunk_size,
            "rank_within_stage": self.rank_within_stage
        }

        self.chunks = math.ceil(self.batch_size / self.micro_batch_size)
        self.schedule = utils.generate_schedule(self.chunks, self.stage, self.partitions)
        self.iteration = 0
        self.current_step = 0

    def init_communication(self):
        rank_within_stage = self.rank_within_stage
        self.send_rank = None; self.receive_rank = None

        # send ranks
        if self.stage < (self.partitions-1):
            self.send_rank = self.stage_to_rank_map[self.stage + 1][rank_within_stage]

        # receive ranks
        if self.stage > 0:
            self.receive_rank = self.stage_to_rank_map[self.stage - 1][rank_within_stage]

        # set expected shapes of inputs and gradients for each partition
        # TODO: are we planning to support multiple acts in cutpoints?
        self.fwd_inp_shape = self.bwd_grad_shape = None
        self.fwd_inp_shape_changes = self.bwd_grad_shape_changes = None
        if self.stage > 0:
            self.fwd_inp_shape = self.model.forward_input_shapes[0]
            self.fwd_inp_shape_changes = self.model.fwd_inp_shape_changes[0]
            for i in self.fwd_inp_shape_changes:
                self.fwd_inp_shape[i] =  self.micro_batch_size
        if self.stage < (self.partitions-1):
            self.bwd_grad_shape = self.model.backward_grad_shapes[0]
            self.bwd_grad_shape_changes = self.model.bwd_grad_shape_changes[0]
            for i in self.bwd_grad_shape_changes:
                self.bwd_grad_shape[i] = self.micro_batch_size

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
        self.tied_group = None
        self.pipeline_group = None
        pipeline_groups = {}
        tied_groups = {}
        for replica in range(self.data_depth):
            ranks = [self.stage_to_rank_map[i][replica] for i in range(self.partitions)]
            if len(ranks) > 1:
                pipeline_groups[replica] = dist.new_group(ranks=ranks)
                if self.shared_weight_stages is not None:
                    recv_stage, send_stage = self.shared_weight_stages[0]
                    tied_ranks = [ranks[recv_stage], ranks[send_stage]]
                    tied_groups[replica] = dist.new_group(ranks=tied_ranks)
                else:
                    tied_groups[replica] = None
            else:
                pipeline_groups[replica] = None
                tied_groups[replica] = None
            
        current_replica = self.stage_to_rank_map[self.stage].index(self.rank)
        print("this rank ", self.rank, "is part of pipeline replica ", current_replica)
        if pipeline_groups[current_replica] is not None:
            self.pipeline_group = pipeline_groups[current_replica]
            self.tied_group = tied_groups[current_replica]

    def configure_checkpointing(self):
        self.param_name_to_pstage = self.partitioned_model.parameter_names_to_cuts()
        # make temp dir for local ckpt trackers
        if self.local_rank == 0 and not os.path.exists(utils.VARUNA_TEMP_FOLDER):
            os.makedirs(utils.VARUNA_TEMP_FOLDER)

    def forward(self, inputs):
        raise RuntimeError("Varuna uses the 'step' function for both fwd/bwd together,\
                             or the 'evaluate' function for evaluation.")

    def step(self, inputs, clip_grad_max_norm=None):
        r""" Perform a single training step. Executes forward and backward passes for 
        the global batch. This function must be called by all distributed workers in the training loop.
        After this function, the optimizer gradients are reduced accross data parallel replicas and
        overflow is checked for mixed precision training. Returns average loss and a boolean for overflow.

        :param inputs: The inputs to the model as a dictionary. These should be coordinated amongst workers -
            the global batch is sharded across data parallel replicas, so each worker should have 
            ``global_batch_size / data_parallel_depth`` number of examples. And all pipeline stages of the same
            data parallel replica should recieve the same inputs.
        :type inputs: dict
        :param clip_grad_max_norm: If given, the L2 gradient norm of the entire model
            is clipped to this upper bound.
        :type clip_grad_max_norm: float or None, optional
        :return: A tuple of the form (average_loss, overflow)
        :rtype: tuple[float, bool]
        """
        assert isinstance(inputs, dict), "Varuna inputs should be a dictionary!"

        # if self.fp16:
        assert self.optimizer is not None, "You must set the optimizer using set_optimizer()"        
        
        # Divide a mini-batch into micro-batches.
        batches = utils.scatter(inputs, int(self.batch_size),self.micro_batch_size)
        
        self.config["make_logfile"] = bool(self.config["make_logfile"] and self.current_step < 5)
        batch_time = time.time()

        self.pipeline = Pipeline(batches, self.model, self.config, self.schedule, self.optimizer, verbose=log_verbose)
        self.average_loss, fwd_time = self.pipeline.run()

        if log_verbose:
            print(f'{self.stage} {self.rank_within_stage} going to share embedding grads')
        
        if self.shared_weights is not None:
            embed_comm_start = time.time()
            self.share_weight_grads()
            embed_comm_time = time.time() - embed_comm_start
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} shared embedding grads')
            print(f'{self.rank} {self.rank_within_stage} all-reduce')

        sync_start_time = time.time()
        if self.fp16 or (self.data_depth > 1) or (self.partitions > 1):
            overflow, grad_norm = self.sync_across_workers(clip_grad_max_norm)
        else:
            overflow = False; grad_norm = 1
        sync_time =  time.time() - sync_start_time
    
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} all-reduce done;')

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

        return self.average_loss, overflow, grad_norm

    def get_status(self):
        return self.pipeline.status

    def get_loss_scale(self):
        if not self.fp16:
            return None
        scaler = _amp_state.loss_scalers[0]
        loss_scale = scaler.loss_scale()
        return loss_scale    

    def evaluate(self, inputs, batch_size=None):
        r"""Evaluate the model on the given inputs. This must be called on all workers
        because it uses pipeline & data parallelism. Inputs should be for the respective data parallel replica
        and have ``batch_size/data_parallel_depth`` examples, similar to :func:`step`.
        Returns loss averaged over all workers.

        :param inputs: Model inputs as dictionary. The number of examples
            for these inputs should be the same as the batch_size defined for training.
        :type inputs: dict
        :param batch_size: Batch size for evaluation, if not given it's the same as training batch size.
        :type batch_size: int, optional
        :return: average loss
        :rtype: float
        """
        assert isinstance(inputs, dict), "input must be a dictionary!"

        if batch_size is None:
            batch_size = self.batch_size
        
        fwd_inp_shape = None
        if self.fwd_inp_shape is not None:
            fwd_inp_shape = list(self.fwd_inp_shape)
            fwd_inp_shape[0] = self.micro_batch_size
        
        batches = utils.scatter(inputs, int(batch_size),self.micro_batch_size)
        
        with torch.no_grad():
            avg_output = None
            chunks = len(batches)
            for i, mb in enumerate(batches):
                if i==(chunks-1) and self.last_chunk_size > 0 and fwd_inp_shape is not None:
                    fwd_inp_shape[0] = self.last_chunk_size
                
                if self.stage > 0:
                    self.partitioned_model.set_recv_acts(fwd_inp_shape, self.receive_rank)
                output = self.partitioned_model(mb)
                if self.stage < self.partitions-1:
                    dist.send(output.cpu(), self.send_rank)

                avg_output = output if avg_output is None else avg_output + output
            self.partitioned_model.clear_recv_fn()
            
        if self.stage == self.partitions - 1:
            output = avg_output / len(batches)
        else:
            output = 0
        output = torch.Tensor([output])
        torch.distributed.all_reduce(output)
        loss = output.item() / self.data_depth     # only last stage on each replica returns >0 loss

        return loss

    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()

    def set_optimizer(self, optimizer, loss_scale = "dynamic",
                            init_loss_scale = 2**20, min_loss_scale=1.0):
        r"""Configure optimizer for training. if ``fp16`` is enabled, this function
        initializes the mixed precision state in apex.

        :param optimizer: the optimizer for training.
        :type optimizer: torch.nn.Optimizer
        :param loss_scale: A floating point number for a static loss scale 
            or the string "dynamic" for dynamic loss scaling.
        :type loss_scale: float or "dynamic", optional
        :param init_loss_scale: Initial loss scale (for dynamic scaling)
        :type init_loss_scale: float, optional
        :param min_loss_scale: minimum loss scale (for dynamic scaling)
        :type min_loss_scale: float, optional
        """
        amp_opt_level="O2"
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
        self.optimizer.zero_grad()
        if self.fp16:
            for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
        for param in self.model.parameters():
            param.grad = None

    def checkpoint(self, global_store, step=None, tempdir=None, shard=False, on_demand=False):
        r""" Writes a varuna checkpoint with model parameters, optimizer state etc. 
        Each checkpoint is a directory, written under the given path.
        
        :param global_store: path to a folder accessible by all nodes/ranks in the training job. 
            For example, path to a mounted blob storage. This is where the varuna checkpoint folder is written.
        :type global_store: dict
        :param step: iteration number for checkpoint. If None, it'll be taken from varuna's tracked progress.
        :type step: int or None, optional
        :param tempdir: path to a local directory to which to write checkpoints temporarily, and sync
            with the global store in the background. Lowers checkpoint write time in the critical path.
        :type tempdir: str, optional
        :param shard: whether to shard checkpoint writes over data parallel workers as well. Speeds up checkpoint 
        :type shard: bool, optional
        """
        if step is None:
            step = self.iteration

        ckpt_future = write_varuna_checkpoint(self, global_store, step, 
                                tempdir=tempdir, shard=shard)
        
        return ckpt_future
    
    def to(self, device):
        self.model.to(device)

    def load_checkpoint(self, global_store, iteration, check_complete = True):
        r"""Loads a varuna checkpoint from a shared directory. Each varuna checkpoint is a directory
        named as "varuna_ckpt_<iteration>". So the path under which all such checkpoints were written
        should be specified.
            
        :param global_store: path under which varuna checkpoints were written. 
            Should be accessible by all workers.
        :type global_store: str
        :param iteration: Which iteration checkpoint to load.
        :type iteration: int
        :param check_complete: Check that the checkpoint is complete before loading it.
            A checkpoint can be incomplete if the write was interrupted.  
        :type check_complete: bool, optional 
        """
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

        # TODO: this should be strict and should raise error in the lm_head_weight case
        self.partitioned_model.module.load_state_dict(model_state_dict)

        load_varuna_optimizer(self.optimizer, self.stage, self.partitions, 
                              total_num_pstages, self.parameter_names, 
                              cp_dir_name, device=self.device)
        # reload master params for mixed precision
        if self.fp16:
            for p in amp.master_params(self.optimizer):
                name = self.parameter_names[p]
                if name in model_state_dict:
                    p.data.copy_(model_state_dict[name].data)

        with open(get_local_ckpt_tracker(self.local_rank),"w") as f:
            # print("writing", iteration)
            f.write(str(iteration))

        self.iteration = iteration    

    def share_weight_grads(self):
        parameter_names = self.parameter_names
        rank_within_stage = self.rank_within_stage
        for i,w in enumerate(self.shared_weights):
            recv_stage, send_stage = self.shared_weight_stages[i]
            recv_wt_name, send_wt_name = w
            recv_weight, send_weight = None, None
            for p in parameter_names:
                if parameter_names[p] == send_wt_name:
                    send_weight = p
                if parameter_names[p] == recv_wt_name:
                    recv_weight = p
            if recv_stage == send_stage and self.stage == recv_stage:
                if recv_weight.data_ptr() != send_weight.data_ptr():
                    recv_weight.grad.add_(send_weight.grad)
                    send_weight.grad.data.copy_(recv_weight.grad.data)
                continue
            if self.stage == send_stage:
                dist.all_reduce(send_weight.grad.data, group=self.tied_group)
            elif self.stage == recv_stage:
                dist.all_reduce(recv_weight.grad.data, group=self.tied_group)

    def all_reduce_dp_grads(self, params):
        allred_init_start = time.time()
        master_grads = [p.grad for p in params if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        flat_raw = torch.empty( flat_grad_size, device=self.device, 
                                dtype=torch.float16 if self.fp16 else torch.float32)

        if self.fp16:        
            scaler = _amp_state.loss_scalers[0]
            loss_scale = scaler.loss_scale()
        else:
            loss_scale = 1

        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf = torch.cuda.IntTensor([0])
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (self.data_depth))

        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} starting gradient all-reduce')

        if self.data_parallel:
            allred_time_start = time.time()
            torch.distributed.all_reduce(flat_raw, group=self.dp_group)
        
        if log_verbose:
            print(f'{self.rank} {self.rank_within_stage} gradient all-reduce done')
            
        if self.fp16:
            amp_C.multi_tensor_scale(65536,
                overflow_buf,
                [allreduced_views, master_grads],
                1./loss_scale)

        return master_grads, overflow_buf

    def sync_across_workers(self, max_norm):

        if self.fp16:
            params = list(amp.master_params(self.optimizer))
        else:
            params = []
            for group in self.optimizer.param_groups:
                params.extend(group['params'])
        
        master_grads, overflow_buf = self.all_reduce_dp_grads(params)

        overflow_buf = overflow_buf.to(torch.float32)
        if overflow_buf.item():
            print(f"{self.rank} Overflow !!")
        overflow_buf, global_grad_norm, reduced_loss = self.all_reduce_pipeline_meta(master_grads, 
                                                                    overflow_buf if self.fp16 else None)
        global_grad_norm_sq = global_grad_norm ** 2
        self.average_loss = reduced_loss

        if max_norm is not None:
            clipped = utils.clip_grad_norm(params, global_grad_norm_sq, max_norm)
            if clipped:
                global_grad_norm = max_norm

        had_overflow = False
        if self.fp16:
            scaler = _amp_state.loss_scalers[0]
            overflow_buf = torch.cuda.IntTensor([0 if overflow_buf.item()==0 else 1])
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overflow_buf = old_overflow_buf

        return had_overflow, global_grad_norm


    """ reduces overflow, norm and loss across pipeline stages """
    def all_reduce_pipeline_meta(self, master_grads, overflow_buf=None):
        
        local_grad_norm = multi_tensor_applier(amp_C.multi_tensor_l2norm,
                                             torch.cuda.IntTensor([0]),
                                             [master_grads], False)[0]
        
        local_grad_norm_sq = (local_grad_norm ** 2) - self.extra_grad_norm_sq()

        loss_tensor = torch.Tensor([self.average_loss]).to(self.device)

        if self.partitions > 1:
            osync_time_start = time.time()
            allred_tensor = torch.cat((local_grad_norm_sq, loss_tensor))
            if overflow_buf is not None:
                allred_tensor = torch.cat((overflow_buf, allred_tensor))
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} starting overflow all_reduce')
            torch.distributed.all_reduce(allred_tensor, group=self.pipeline_group)
            if log_verbose:
                print(f'{self.rank} {self.rank_within_stage} overflow all_reduce done')
            
            if overflow_buf is not None:
                overflow_buf, global_grad_norm_sq, loss_tensor = allred_tensor
            else:
                global_grad_norm_sq, loss_tensor = allred_tensor
            global_grad_norm = global_grad_norm_sq ** 0.5

        else:
            global_grad_norm_sq = local_grad_norm_sq
            global_grad_norm = global_grad_norm_sq ** 0.5

        return overflow_buf, global_grad_norm, loss_tensor.item()
        
    def extra_grad_norm_sq(self):
        extra_norm_sq = 0.0
        if self.shared_weights is None:
            return extra_norm_sq
        for i,w in enumerate(self.shared_weights):
            recv_stage, send_stage = self.shared_weight_stages[i]
            recv_wt_name, send_wt_name = w
            recv_weight, send_weight = None, None
            for p in self.parameter_names:
                if self.parameter_names[p] == send_wt_name:
                    send_weight = p
                if self.parameter_names[p] == recv_wt_name:
                    recv_weight = p
            if recv_stage == send_stage:
                if self.stage == recv_stage and recv_weight.data_ptr() == send_weight.data_ptr():
                    continue
            if self.stage == send_stage:
                if send_weight.grad is not None:
                    extra_norm_sq += torch.norm(send_weight.grad) ** 2

        return extra_norm_sq
