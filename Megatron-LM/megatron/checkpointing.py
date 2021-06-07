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

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np
import shutil

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import mpu
from megatron import get_args
from megatron import print_rank_0
import concurrent.futures

from varuna import get_this_rank_config_varuna

from apex import amp


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retreived frm checkpoint."""
    args = get_args()

    def _compare(arg_name):
        checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('max_position_embeddings')
    _compare('make_vocab_size_divisible_by')
    _compare('padded_vocab_size')
    _compare('tokenizer_type')
    _compare('model_parallel_size')


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration, on_demand=False, dp_rank=0,
                        release=False, mp_rank=None):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    if mp_rank is None:
        mp_rank = mpu.get_model_parallel_rank() if mpu.model_parallel_is_initialized() else 0
    filename = 'model_optim_rng.pt'
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}'.format(mp_rank),
                        filename)


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer, lr_scheduler, on_demand=False, bgd=True):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    if not args.varuna and isinstance(model, torchDDP):
        model = model.module

    if args.varuna:
        _, data_parallel_rank = get_this_rank_config_varuna(args.stage_to_rank_map, args.rank)
    elif mpu.model_parallel_is_initialized():
        data_parallel_rank = mpu.get_data_parallel_rank()
    else:
        data_parallel_rank = torch.distributed.get_rank()
    
    if on_demand or data_parallel_rank == 0:

        tempdir = "/mnt/nitika/varuna_ckpts/"
        checkpoint_name = get_checkpoint_name(args.save, iteration, data_parallel_rank)

        if args.rank == 0:
            ensure_directory_exists(checkpoint_name)
        if args.local_rank == 0:
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)

        # Arguments, iteration, and model.
        if args.rank == 0:
            state_dict = {}
            state_dict['args'] = args
            state_dict['iteration'] = iteration
            if not args.varuna:
                state_dict['model'] = model.state_dict_for_save_checkpoint()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    opt_state = optimizer.state_dict()
                    if args.varuna:
                        opt_state["state"] = {}
                    state_dict['optimizer'] = opt_state
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            # RNG states.
            if not args.no_save_rng:
                state_dict['random_rng_state'] = random.getstate()
                state_dict['np_rng_state'] = np.random.get_state()
                state_dict['torch_rng_state'] = torch.get_rng_state()
                state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
                state_dict['rng_tracker_states'] \
                    = mpu.get_cuda_rng_tracker().get_states()

            if args.fp16:
                state_dict['amp'] = amp.state_dict()

            # Save.
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration, checkpoint_name))
            torch.save(state_dict, checkpoint_name)

        if args.varuna:
            mv_futures = model.checkpoint(args.save, step=iteration, 
                    tempdir=tempdir if bgd else None, shard = on_demand)
                    
        # remove old checkpoints
        if (not on_demand) and args.max_num_ckpts is not None and torch.distributed.get_rank() == 0:
            all_ckpted_iters = sorted([int(f.split("_")[-1]) for f in os.listdir(args.save) if f.startswith("varuna_ckpt")])
            # assert all_ckpted_iters[-1] == iteration, "The latest checkpoint is corrupted?"
            if len(all_ckpted_iters) > args.max_num_ckpts:
                to_remove = all_ckpted_iters[:-args.max_num_ckpts]
                print("removing older checkpoints at: ", to_remove)
                for it in to_remove:
                    if it <= args.min_ckpt_iter_to_remove:
                        continue
                    try:
                        os.system("rm -rf {} &".format(os.path.join(args.save,"varuna_ckpt_{}".format(it))))
                        os.system("rm -rf {} &".format(os.path.join(args.save,'iter_{:07d}'.format(it))))
                    except Exception as e:
                        print("Error while removing checkpoint {}: {}".format(it,str(e)))
        print('  successfully saved {}'.format(checkpoint_name))

    torch.distributed.barrier()


def parse_last_ckpt_iteration():

    args = get_args()
    if args.load_iteration != -1:
        return args.load_iteration, False

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(args.load)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                    'random')
        return 0, False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release


def load_checkpoint(model, optimizer, lr_scheduler):
    """Load a model checkpoint and return the iteration."""
    args = get_args()

    if isinstance(model, torchDDP):
        model = model.module

    iteration, release = parse_last_ckpt_iteration()

    if iteration == 0:
        return 0
        
    # Checkpoint.
    checkpoint_name = get_checkpoint_name(args.load, iteration, release)
    if (not mpu.model_parallel_is_initialized()) or mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        # For backward compatibility.
        print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16.loss_scaler']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
    except BaseException:
        print_rank_0('could not load the checkpoint')
        sys.exit()

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()

    # Check arguments.
    if 'args' in state_dict:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if args.varuna:
        model.load_checkpoint(args.load, iteration)
    else:
        model.load_state_dict(state_dict['model'])

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                opt_dict = state_dict['optimizer']
                if not args.varuna:
                    optimizer.load_state_dict(opt_dict)
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    if args.fp16:
        amp.load_state_dict(state_dict['amp'])
    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    torch.distributed.barrier()
    print('  successfully loaded {}'.format(checkpoint_name))

    return iteration
