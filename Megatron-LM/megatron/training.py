# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import os
import time

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedLAMB as LAMB

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_preempt_signal
from megatron import mpu
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint, parse_last_ckpt_iteration
from megatron.checkpointing import save_checkpoint
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import make_data_loader
from megatron.utils import report_memory
from megatron.utils import get_ltor_masks_and_position_ids

from varuna import Varuna, PartitionedModel
from varuna import load_varuna_checkpoint

from apex import amp
from apex.amp import _amp_state

accumulated_loss = 0

def pretrain(train_valid_test_dataset_provider, model_provider,
             forward_step_func, eval_step_varuna, extra_args_provider=None, args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    args = get_args()
    timers = get_timers()

    if args.load is not None:
        args.iteration, _ = parse_last_ckpt_iteration()
    else:
        args.iteration = 0

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator, dry_run_input \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()

    if args.varuna:
        device = args.local_rank

        # Unpack.
        tokens_ = torch.Tensor(dry_run_input['text']).view(1,-1).long()
        labels = tokens_[:,1:].contiguous()
        tokens = tokens_[:,:-1].contiguous()

        # Get the masks and postition ids.
        tokenizer = get_tokenizer()
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        dry_run_input = dict({
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": labels
        })
        
    # Model, optimizer, and learning rate.    
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler, parameter_names = setup_model_and_optimizer(model_provider, dry_run_input if args.varuna else None)
    timers('model and optimizer').stop()
    
    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        if args.do_train:
            iteration, _ = train(forward_step_func,
                                 model, optimizer, lr_scheduler,
                                 train_data_iterator, valid_data_iterator, parameter_names, eval_step_varuna)

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func if not args.varuna else eval_step_varuna,
                                   valid_data_iterator, model,
                                   iteration, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler,parameter_names)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func if not args.varuna else eval_step_varuna,
                                   test_data_iterator, model,
                                   0, True)


def get_model(model_provider_func, dry_run_input=None):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    if args.varuna:
        shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]
        model = Varuna(model, args.stage_to_rank_map, dry_run_input, args.batch_size * args.data_depth, args.chunk_size, args.fp16, local_rank=args.local_rank, device=args.local_rank, shared_weights=shared_weights)            
    if args.local_rank == 0:
        print('get_model() post varuna init:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())            
    
    # Print number of parameters.
    if mpu.model_parallel_is_initialized() and  mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)
    else:
        print(args.rank, ": total num params is", sum([p.nelement() for p in model.parameters()]), flush=True)
    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    return model


def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (torchDDP, LocalDDP, FP16_Module)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    if mpu.model_parallel_is_initialized():
        for param_group in param_groups:
            for param in param_group['params']:
                if not hasattr(param, 'model_parallel'):
                    param.model_parallel = False

    # Use LAMB/Adam.
    if args.use_adam:
        optimizer = Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:    
        optimizer = LAMB(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func, dry_run_input=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, dry_run_input)
    optimizer = get_optimizer(model)
    if args.local_rank==0:
        print('setup_model() post optimizer init:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    basemodel = model
    while isinstance(basemodel, (Varuna,PartitionedModel, torchDDP)):
        basemodel = basemodel.module if hasattr(basemodel, "module") else basemodel.model

    if args.fp16:
        if args.dynamic_loss_scale:
            basemodel, optimizer = amp.initialize(basemodel, optimizer, opt_level="O2", loss_scale="dynamic",min_loss_scale=args.min_scale)
            amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        else:
            basemodel, optimizer = amp.initialize(basemodel, optimizer, opt_level="O2", loss_scale=args.loss_scale, min_loss_scale=args.min_scale)
    
    if args.local_rank==0:
        print('setup_model() post amp init:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        if args.varuna:
            model.model.module = basemodel

    if args.varuna:
        model.set_optimizer(optimizer)

    # fp32 param names for checkpointing
    optimizer._amp_lazy_init()
    if args.local_rank==0:
        print('setup_model() post amp opt init:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    parameter_names_ = dict()
    for n,p in basemodel.named_parameters():
        parameter_names_[p] = n
    fp16_model_params = optimizer._amp_stash.all_fp16_params
    fp32_master_params = optimizer._amp_stash.all_fp32_from_fp16_params
    print("stash lens",len(fp16_model_params), len(fp32_master_params))
    count = 0
    parameter_names = dict()
    for p_model, p_master in zip(fp16_model_params, fp32_master_params):
        if p_model in parameter_names_:
            parameter_names[p_master] = parameter_names_.pop(p_model)
            count += 1
    print(count, "params found in rank", args.rank)
    # if args.local_rank==0:
        # print("setup_model() post opt init: ", args.local_rank, torch.cuda.memory_summary(torch.cuda.current_device()))
        # print('setup_model() post opt int:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    # print(args.rank, parameter_names)
    # '''

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        assert i == args.local_rank, "Local rank and device should be same!"
        process_group = mpu.get_data_parallel_group() if mpu.model_parallel_is_initialized() else None
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=process_group)
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        
    if args.load is not None:
        args.iteration = load_checkpoint(basemodel, optimizer, lr_scheduler, parameter_names)
    else:
        args.iteration = 0

    # this needs to be fixed asap
    if args.varuna and args.stage == args.partitions - 1:
        param = None
        for p in parameter_names:
            if parameter_names[p] == "lm_head_weight":
                param = p
                break
        with torch.no_grad():
            basemodel.lm_head_weight.data.copy_(param.data) 


    return model, optimizer, lr_scheduler, parameter_names


def backward_step(optimizer, model, loss, iteration):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    loss = loss / args.gradient_accumulation_steps

    if args.fp16:
        with amp.scale_loss(loss, optimizer, delay_overflow_check=False, last_microbatch=bool((iteration+1) % args.gradient_accumulation_steps == 0), last_partition=True) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0 and (iteration+1) % args.gradient_accumulation_steps == 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            mpu.clip_grad_norm(amp.master_params(optimizer), args.clip_grad)


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler, iteration):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Forward model for one step.
    timers('forward').start()
    if isinstance(model,torchDDP) and ((iteration+1) % args.gradient_accumulation_steps != 0):
        with model.no_sync():
            loss, loss_reduced = forward_step_func(data_iterator, model)
    else:
        loss, loss_reduced = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss, iteration)
    timers('backward').stop()

    skipped_iter = 0
    if (iteration+1) % args.gradient_accumulation_steps == 0:
        # Update parameters
        timers('optimizer').start()
        overflow = optimizer.step()
        timers('optimizer').stop()
        for param in model.parameters():
            param.grad = None

        # Update learning rate.
        if not (args.fp16 and overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1

    return loss_reduced, skipped_iter

def train_step_varuna(varuna_step, data_iterator,model, optimizer, lr_scheduler, iteration):
    """Single training step varuna."""
    args = get_args()
    timers = get_timers()

    # Forward model for one step.
    # timers('forward').start()
    loss, loss_reduced, overflow = varuna_step(data_iterator, model)
    # timers('forward').stop()

    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            mpu.clip_grad_norm(amp.master_params(optimizer), args.clip_grad)

    # Update parameters.
    timers('optimizer').start()
    # if args.local_rank==0:
        # print("setup_model() pre opt step: ", args.local_rank, torch.cuda.memory_summary(torch.cuda.current_device()))
        # print('setup_model() pre opt step:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    if not overflow:
        optimizer.step()
    else:
        for param in optimizer._amp_stash.all_fp32_from_fp16_params:
            param.grad = None
    # if args.local_rank==0:
        # print("setup_model() post opt step: ", args.local_rank, torch.cuda.memory_summary(torch.cuda.current_device()))
        # print('setup_model() post opt step:', args.local_rank, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())        
    timers('optimizer').stop()

    for param in model.parameters():
        param.grad = None

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return loss_reduced, skipped_iter


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, step_time, total_train_time, report_memory_flag, loss_file=None):
    global accumulated_loss
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Update losses.
    for key in loss_dict:
        total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward')
    add_to_logging('backward')
    add_to_logging('allreduce')
    add_to_logging('optimizer')
    add_to_logging('batch generator')

    # Tensorboard values.
    should_log = (args.stage == args.partitions - 1) if args.varuna \
        else torch.distributed.get_rank() == 0
    if writer and should_log:
        writer.add_scalar('learning_rate', learning_rate, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
        if args.fp16:
            writer.add_scalar('loss_scale', loss_scale, iteration)
        normalizer = iteration % args.log_interval
        if normalizer == 0:
            normalizer = args.log_interval
        timers.write(timers_to_log, writer, iteration,
                     normalizer=normalizer)

    complete_steps = iteration // args.gradient_accumulation_steps
    lm_loss = loss_dict["lm loss"].item() if isinstance(loss_dict["lm loss"], torch.Tensor) else loss_dict["lm loss"]
    accumulated_loss += lm_loss
    if (loss_file is not None) and iteration % args.gradient_accumulation_steps == 0:
        accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
        loss_file.write("{}, {}, {}, {}, {}\n".format(step_time, total_train_time, loss_scale, learning_rate, accumulated_loss))
        if complete_steps % 50:
            loss_file.flush()
        accumulated_loss = 0

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval time').elapsed()
        if writer and should_log:
            writer.add_scalar('iteration_time',
                              elapsed_time / args.log_interval, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                       args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time * 1000.0 / args.log_interval)
        log_string += 'total train time(s): {:.1f}'.format(total_train_time)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        for key in total_loss_dict:
            avg = total_loss_dict[key].item() / args.log_interval
            log_string += ' {}: {:.6E} |'.format(key, avg)
            total_loss_dict[key] = 0.0
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if should_log:
            print(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator, parameter_names, eval_step_varuna=None):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    if args.debug_eval:
        model.eval()
    else:
        model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration * args.gradient_accumulation_steps
    skipped_iters = 0
    complete_steps = args.iteration
    if args.varuna:
        model.step = args.iteration

    loss_file = None
    eval_loss_file = None
    if (args.varuna and args.stage == args.partitions - 1) or (not args.varuna and torch.distributed.get_rank() == 0):
        loss_filename = "{}-{}.txt".format(args.loss_file, torch.distributed.get_rank() )
        eval_loss_filename = "eval-" + loss_filename
        loss_filename = os.path.join(args.save, "stats", loss_filename)
        eval_loss_filename = os.path.join(args.save, "stats", eval_loss_filename)
        if iteration == 0:
            if os.path.isfile(loss_filename):
                raise RuntimeError("loss file {} already exists!".format(loss_filename))
            loss_file = open(loss_filename,"w")
            loss_file.write("Loss scale, loss\n")
            eval_loss_file = open(eval_loss_filename, "w")
            eval_loss_file.write("Iteration, loss keys")
        else:
            loss_file = open(loss_filename,"a")
            loss_file.write("resumed to config {}x{} at step {}\n".format(args.partitions, args.data_depth, args.iteration))
            eval_loss_file = open(eval_loss_filename, "a")
            eval_loss_file.write("resumed\n")
        loss_file.write(str(datetime.now())+ "\n")
    
    timers('interval time').start()
    report_memory_flag = True

    train_start_time = time.time()

    step_func = train_step_varuna if args.varuna else train_step

    while iteration < args.train_iters:
        step_time = time.time()
        loss_dict, skipped_iter = step_func(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler,
                                             iteration)
        step_time = time.time() - step_time
        skipped_iters += skipped_iter
        iteration += 1
        if iteration % args.gradient_accumulation_steps == 0:
            complete_steps += 1

        # Logging.
        total_train_time = time.time() - train_start_time
        loss_scale = None
        if args.fp16:
            loss_scale = _amp_state.loss_scalers[0].loss_scale()
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale, step_time, total_train_time,
                                          report_memory_flag, loss_file)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Checkpointing
        if args.save and args.save_interval and complete_steps > 0 and \
           complete_steps % args.save_interval == 0:
            ckpt_time = time.time()           
            save_checkpoint(complete_steps, model, optimizer, lr_scheduler, parameter_names)
            ckpt_time = time.time() - ckpt_time
            print(args.rank, "ckpt time", ckpt_time)
            
        # Evaluation
        if args.eval_interval and complete_steps > 0 and complete_steps % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(complete_steps)
            evaluate_and_print_results(prefix, forward_step_func if not args.varuna else eval_step_varuna,
                                       valid_data_iterator, model,
                                       complete_steps, False, eval_loss_file)

        if args.exit_interval and complete_steps > 0 and complete_steps % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, complete_steps))
            sys.exit()

    if loss_file is not None:
        loss_file.close()
    return complete_steps, skipped_iters


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))
            # Forward evaluation.
            _, loss_dict = forward_step_func(data_iterator, model)
            # Reduce across processes.
            if not args.varuna or args.stage == args.partitions - 1:
                for key in loss_dict:
                    total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                        loss_dict[key]
    
    # Move model back to the train mode.
    if not args.debug_eval:
        model.train()

    if not args.varuna or args.stage == args.partitions - 1:
        for key in total_loss_dict:
            total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False, loss_file=None):
    """Helper function to evaluate and dump results on screen."""
    writer = get_tensorboard_writer()
    args = get_args()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)

    # print results
    should_log = (args.stage == args.partitions - 1) if args.varuna \
        else torch.distributed.get_rank() == 0
    if should_log:
        if loss_file is not None:
            loss_file.write(str(iteration) + ": ")
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} value'.format(key),
                            total_loss_dict[key].item(),
                            iteration)
            writer.add_scalar('{} ppl'.format(key), ppl, iteration)
        if loss_file is not None:
            loss_file.write("{}: {}, ".format(key, total_loss_dict[key].item()))

    if loss_file is not None:
        loss_file.write("\n")
 
    if should_log:
        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if (not mpu.model_parallel_is_initialized()) or mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        if args.varuna:
            data_parallel_size = args.data_depth
        elif mpu.model_parallel_is_initialized():
            data_parallel_size = mpu.get_data_parallel_world_size()
        else:
            data_parallel_size = torch.distributed.get_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        dry_run_input = train_ds[0]

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if mpu.model_parallel_is_initialized():
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator, dry_run_input


def get_eval_numbers(train_valid_test_dataset_provider, model_provider,
             eval_step_varuna):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args = get_args()

    print("WS",args.world_size, torch.distributed.get_world_size())

    shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]
    model, _opt, _lrs, _pn = setup_model_and_optimizer(model_provider, None)
    model = Varuna(model, args.stage_to_rank_map, {}, args.batch_size * args.data_depth, _opt, args.chunk_size, args.fp16, local_rank=args.local_rank, device=args.local_rank, shared_weights=shared_weights)            

    ckpt_iters = sorted([int(f.split("_")[-1]) for f in os.listdir(args.load) if "model_ckpt_" in f])
    ckpts = ["model_ckpt_{}".format(i) for i in ckpt_iters]
    print(ckpts)
    
    eval_samples = len(ckpts) * args.batch_size * args.eval_iters
    _tr, valid_ds, _te = train_valid_test_dataset_provider([0,eval_samples,0])

    valid_dataloader = make_data_loader(valid_ds)
    valid_data_iterator = iter(valid_dataloader)

    
    loss_file = None
    if args.stage == args.partitions - 1:
        loss_file = open("eval_loss_varuna_350m_32k.txt", "w")

    for model_ckpt in ckpts:
        iteration = int(model_ckpt.split("_")[-1])
        print("Evaluating checkpoint", iteration)

        model_state_dict = load_varuna_checkpoint(args.stage, args.partitions, args.num_layers, os.path.join(args.load,model_ckpt))
        model.model.module.load_state_dict(model_state_dict, strict = False)
        print("loaded checkpoint!")
        opt_state_dict = torch.load(os.path.join(args.load,"opt_ckpt_{}/opt-fp32-params-{}".format(iteration, args.num_layers-1)), map_location="cpu")
        lm_head_weight = opt_state_dict["lm_head_weight"].to(args.local_rank)
        if args.stage == 0:
            model.model.module.language_model.embedding.word_embeddings.weight.data.copy_(lm_head_weight.data)
        if args.stage == args.partitions - 1:
            model.model.module.lm_head_weight.data.copy_(lm_head_weight.data)
        total_loss_dict = evaluate(eval_step_varuna, valid_data_iterator, model, verbose=True)
        if args.stage == args.partitions - 1:
            loss_file.write(str(iteration) + " ")
            for key in total_loss_dict: 
                loss = total_loss_dict[key].item()
                ppl = math.exp(min(20, loss))
                if loss_file is not None:
                    loss_file.write("{}: {}, {};".format(key, loss, ppl))

        if loss_file is not None:
            loss_file.write("\n")

