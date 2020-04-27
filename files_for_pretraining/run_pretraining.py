# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp
import multiprocessing
from collections import defaultdict

from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig
from apex.optimizers import FusedLAMB
from schedulers import PolyWarmUpScheduler

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process
from apex.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler
from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state
import signal

from concurrent.futures import ProcessPoolExecutor

from varuna import Varuna, load_varuna_checkpoint
import datetime

TERMINATE_TRAINING = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, my_stage_ranks):

    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)

    # Each stage divides data according to it's depth
    if len(my_stage_ranks) > 1:
        num_replicas = len(my_stage_ranks)
        rank_within_stage = my_stage_ranks.index(args.rank)
        print(args.rank, "has stage with ranks", my_stage_ranks,"; index", rank_within_stage)
        train_sampler = DistributedSampler(train_data, num_replicas=num_replicas, rank=rank_within_stage, shuffle=False)
        train_dataloader = DataLoader(train_data, batch_size=(args.train_batch_size // num_replicas), sampler=train_sampler, shuffle = False, drop_last = True, pin_memory=True, num_workers=4)
    else:
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                    batch_size=args.train_batch_size, num_workers=4,
                                    pin_memory=True, shuffle = False, drop_last = True)

    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        # print("getting index ", index, masked_lm_positions)

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--device",
                        type=int,
                        default=-1, 
                        help="GPU number to use for compute")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=5,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    # arguments for Varuna
    parser.add_argument('--chunk_size',
                        type=int,
                        default=1,
                        help="Micro batch size per per mini-batch")
    parser.add_argument('--partitions',
                        type=int,
                        default=1,
                        help="Number of devices over which the model is split")
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help="Partition index")
    parser.add_argument("--stage_to_rank_map",
                        type=str,
                        default="",
                        help="Stage to rank map for pipeline")
    args = parser.parse_args()
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    # '''
    if args.device == -1:
        args.device = args.local_rank
    args.n_gpu = 1
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    connect_timeout = datetime.timedelta(minutes=10)
    world_size = len(args.stage_to_rank_map.split(";")[0].split(",")) * args.partitions
    torch.distributed.init_process_group(backend='gloo', timeout=connect_timeout, world_size=world_size, rank=args.rank)
    # '''
    '''
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("device %s n_gpu %d distributed training %r", device, args.n_gpu, bool(args.local_rank != -1))
    # '''

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    # if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
    #         os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # if not args.resume_from_checkpoint:
    #     os.makedirs(args.output_dir, exist_ok=True)
    # '''

    return device, args

def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
            model_cp_dir = os.path.join(args.output_dir, "model_ckpt_{}".format(global_step))
            print("loading varuna ckpt", global_step)
            model_state_dict = load_varuna_checkpoint(args.stage, args.partitions, config.num_hidden_layers, model_cp_dir)
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")

        model.load_state_dict(model_state_dict, strict = False)
        if args.phase2:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters, 
                          lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                       warmup=args.warmup_proportion, 
                                       total_steps=args.max_steps)
    
    # this map from optimizer parameters to their names is to checkpoint opt state
    parameter_names = dict()
    for n,p in model.named_parameters():
        parameter_names[p] = n
    
    if args.fp16:

        if args.loss_scale == 0:# and args.rank == args.partitions-1:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
            amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=8.0)

        # need to check if parameter_names map works for new fp16 model params
        for n,p in model.named_parameters():
            assert parameter_names[p] == n, "this is wrong"

        # creating new fp32 params for mixed precision optimizer
        # and mapping these parameters to their names
        optimizer._amp_lazy_init()
        fp16_model_params = optimizer._amp_stash.all_fp16_params
        fp32_master_params = optimizer._amp_stash.all_fp32_from_fp16_params
        for p_model, p_master in zip(fp16_model_params, fp32_master_params):
            parameter_names[p_master] = parameter_names.pop(p_model)
        # parameter_names = parameter_names_

    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        
        amp.load_state_dict(checkpoint['amp'])
        # reload optimizer state
        opt_dict = checkpoint['optimizer']
        opt_cp_dir = os.path.join(args.output_dir, "opt_ckpt_{}".format(global_step))
        optimizer.load_state_dict(opt_dict)  # , strict=False)
        optimizer.state = defaultdict(dict)
        opt_state = {}
        state_filenames = [f for f in os.listdir(opt_cp_dir) if "opt-state-" in f]
        if len(state_filenames) == args.partitions:
            opt_state = torch.load(os.path.join(opt_cp_dir,"opt-state-{}".format(args.stage)))
        else:
            for f in state_filenames:
                state_ = torch.load(os.path.join(opt_cp_dir,f),map_location='cpu')
                opt_state.update(state_)
        success = 0
        for p in amp.master_params(optimizer):
            name = parameter_names[p]
            if name in opt_state:
                optimizer.state[p] = opt_state[name]
                success += 1
        print(success, "resume success")

        # Restore AMP master parameters          
        if args.fp16:
            # optimizer._lazy_init_maybe_master_weights()
            # optimizer._amp_stash.lazy_init_called = True
            # optimizer.load_state_dict(checkpoint['optimizer'])
            assert optimizer._amp_stash.lazy_init_called, "should already be true!" 
            master_params = torch.load(os.path.join(opt_cp_dir,"opt-fp32-params-{}".format(args.stage)))
            for param, saved_param in zip(amp.master_params(optimizer), master_params):
                param.data.copy_(saved_param.data)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)


    return model, optimizer, lr_scheduler, checkpoint, global_step, parameter_names

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        # amp_C.multi_tensor_scale(65536,
        #     overflow_buf,
        #     [master_grads, allreduced_views],
        #     scaler.loss_scale() / (torch.distributed.get_world_size() * args.gradient_accumulation_steps))
        amp_C.multi_tensor_scale(65536, overflow_buf, [master_grads, allreduced_views],
            scaler.loss_scale() / args.chunks) #args.gradient_accumulation_steps) 
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        # torch.distributed.all_reduce(flat_raw)              # comment this for Varuna: not needed here
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            # if is_main_process():
            print(("Rank {} :: Gradient overflow.  Skipping step, "  +
                    "reducing loss scale to {}").format(
                    torch.distributed.get_rank(),
                    scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step

def main():

    def handler(signum,_):
        global TERMINATE_TRAINING
        print(args.rank, 'signal handler called with signal', signum)
        TERMINATE_TRAINING = True
    
    signal.signal(signal.SIGUSR1, handler) 

    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # parse stage_to_rank_map
    stage_to_rank_map = args.stage_to_rank_map
    stage_ranks = stage_to_rank_map.split(";", args.partitions)
    stage_to_rank_map = {}
    for i in range(args.partitions):
        ranks = stage_ranks[i].split(",")
        stage_to_rank_map[i] = [int(r) for r in ranks]
    data_depth = len(ranks)

    for stage in stage_to_rank_map:
        for rank in stage_to_rank_map[stage]:
            if rank == args.rank:
                args.stage = stage
                break

    device, args = setup_training(args)

    if args.stage == args.partitions - 1:
        loss_filename = 'stats/varuna_lamb_'+("fp16" if args.fp16 else "fp32") +"_"+str(args.learning_rate)+"_"+str(args.train_batch_size)+'_'+str(args.partitions)+'p_'+str(data_depth)+'dp_'+str(args.chunk_size)+'csize_'+str(args.rank) + '.txt'
        if os.path.isfile(loss_filename):
            if args.resume_from_checkpoint:
                loss_file = open(loss_filename, 'a')
                loss_file.write("resuming\n")
            else:
                raise RuntimeError("File ({}) already exists.".format(loss_filename))
        else:
            loss_file = open(loss_filename, 'w')
            # loss_file.write("MB time, total train time, TFLOPS, Max GPU mem, Curr GPU mem, Opt state mem, loss scale, loss\n")

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, parameter_names = prepare_model_and_optimizer(args, device)

    if is_main_process():
        print("SEED {}".format(args.seed))

    if args.do_train:
        if is_main_process():
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", args.train_batch_size)
            print("  LR = ", args.learning_rate)
            print("Training. . .")

        # model.train()       # comment this for Varuna
        initialize_model=True
        most_recent_ckpts_paths = []
        most_recent_model_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0
        avg_mb_time = 0
        avg_tflops = 0
        train_start_time = time.time()

        pool = ProcessPoolExecutor(1)

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)


            shared_file_list = {}

            '''
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                remainder = torch.distributed.get_world_size() % num_files
                data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_start_id)%num_files]
            else:
                data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]
            # '''
            data_file = files[(f_start_id+0)%num_files]

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)

            my_stage_ranks = stage_to_rank_map[args.stage]

            if len(my_stage_ranks) > 1:
                num_replicas = len(my_stage_ranks)
                rank_within_stage = my_stage_ranks.index(args.rank)
                # print(args.rank, "has stage with ranks", my_stage_ranks,"; index", rank_within_stage)
                train_sampler = DistributedSampler(train_data, num_replicas=num_replicas, rank=rank_within_stage, shuffle=False)
                train_dataloader = DataLoader(train_data, batch_size=(args.train_batch_size // num_replicas), sampler=train_sampler, shuffle = False, drop_last = True, pin_memory=True, num_workers=4)
            else:
                train_sampler = SequentialSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                            batch_size=args.train_batch_size, num_workers=4,
                                            pin_memory=True, shuffle = False, drop_last = True)

            dummy_input = dict()
            dummy_train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1, num_workers=1, pin_memory=False)
            train_iter = tqdm(dummy_train_dataloader, desc="Iteration")
            if initialize_model:
                for step, batch in enumerate(train_iter):
                # batch = train_data[:1]
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    dummy_input['input_ids'] = input_ids
                    dummy_input['token_type_ids'] = segment_ids
                    dummy_input['attention_mask'] = input_mask
                    dummy_input['masked_lm_labels'] = masked_lm_labels
                    dummy_input['next_sentence_label'] = next_sentence_labels
                    break
                del train_iter
                del dummy_train_dataloader

                model = Varuna(model, stage_to_rank_map, dummy_input, args.train_batch_size, optimizer, args.chunk_size, args.fp16, local_rank=args.local_rank, device=args.device)
                model.train()
                initialize_model=False
            # '''

            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])
            
            if len(files) == 1:
                f_start_id = -1
            for f_id in range(f_start_id + 1 , len(files)):
                '''
                if torch.distributed.get_world_size() > num_files:
                    data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_id)%num_files]
                else:
                    data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]
                # '''
                data_file = files[f_id%num_files]
                logger.info("file no %s file %s" % (f_id, previous_file))
                previous_file = data_file
                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, my_stage_ranks)
                train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader


                for step, batch in enumerate(train_iter):
                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    # '''
                    inputs = dict()
                    inputs['input_ids'] = input_ids
                    inputs['token_type_ids'] = segment_ids
                    inputs['attention_mask'] = input_mask
                    inputs['masked_lm_labels'] = masked_lm_labels
                    inputs['next_sentence_label'] = next_sentence_labels

                    torch.cuda.reset_max_memory_allocated(args.device)
                    pre_pipeline_mem = torch.cuda.memory_allocated(args.device)
                    minibatch_time = time.time()
                    loss = model(inputs)
                    minibatch_time = time.time() - minibatch_time
                    tflops = 1.33 * (args.train_batch_size / minibatch_time)     # TFLOPS ~ 1.33 * examples per sec
                    tflops = tflops / (args.partitions * data_depth)        # scale by # gpus
                    avg_mb_time += minibatch_time
                    avg_tflops += tflops
                    
                    average_loss += loss      # comment this for running without Varuna
                    divisor = args.gradient_accumulation_steps  # args.chunks

                    if training_steps % args.gradient_accumulation_steps == 0:
                        lr_scheduler.step()  # learning rate warmup
                        # print(args.rank, "before optimizer step",torch.cuda.memory_allocated(args.device))
                        global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)
                        max_mem = torch.cuda.max_memory_allocated(args.device)
                        curr_mem = torch.cuda.memory_allocated(args.device)
                        opt_state_mem = curr_mem - pre_pipeline_mem

                    if global_step >= args.max_steps:
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()       # comment this for varuna - maybe not, commenting this is what is causing error at the end of epoch
                        average_loss = average_loss / (last_num_steps * divisor)
                        if is_main_process():
                            logger.info("Total Steps:{} Final Loss = {}".format(training_steps / args.gradient_accumulation_steps, average_loss.item()))
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if args.stage == args.partitions - 1:
                            print("Step:{} Mini-batch time = {} Average Loss = {} Step Loss = {} LR {}".format(global_step, minibatch_time, average_loss / (
                                        args.log_freq * divisor), loss * args.gradient_accumulation_steps / divisor,
                                        optimizer.param_groups[0]['lr']))
                        if args.stage == args.partitions - 1:
                            if global_step%10==0:
                                loss_file.flush()
                            total_train_time = time.time() - train_start_time
                            loss_scale = _amp_state.loss_scalers[0].loss_scale()
                            loss_file.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(minibatch_time, total_train_time, tflops, max_mem, curr_mem, opt_state_mem, loss_scale, average_loss))
                            # loss_file.write("{}, ".format(average_loss))

                        average_loss = 0 

                    if global_step >= args.max_steps or TERMINATE_TRAINING or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                            # Save a trained model
                        model_cp_dir = os.path.join(args.output_dir, "model_ckpt_{}".format(global_step))
                        opt_cp_dir = os.path.join(args.output_dir, "opt_ckpt_{}".format(global_step))
                        if args.rank == 0 and not os.path.exists(model_cp_dir):
                            os.makedirs(model_cp_dir)
                        if args.rank == 0 and not os.path.exists(opt_cp_dir):
                            os.makedirs(opt_cp_dir)
                        torch.distributed.barrier()
                        model.checkpoint(model_cp_dir)
                        model.checkpoint_optimizer(optimizer, parameter_names, opt_cp_dir)
                        if args.rank == stage_to_rank_map[args.stage][0]:
                            master_params = list(amp.master_params(optimizer))
                            # assert len(master_params) == 398,"NO! got {} master params".format(len(master_params))
                            torch.save(master_params, os.path.join(opt_cp_dir,"opt-fp32-params-{}".format(args.stage)))
                        torch.distributed.barrier()
                        if args.rank == 0:
                            # assert model_state_dict is not None, "Wrong checkpointing!!"
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                            else:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                            if args.do_train:
                                opt_state_dict = optimizer.state_dict()
                                opt_state_dict["state"] = {}
                                torch.save({'optimizer': opt_state_dict,
                                            # 'model': model_state_dict,
                                            # 'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files,
                                            'amp': amp.state_dict()}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                most_recent_model_ckpts_paths.append(model_cp_dir)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)
                                    # os.rmdir(most_recent_model_ckpts_paths.pop(0))


                    if global_step >= args.max_steps or TERMINATE_TRAINING:
                            del train_dataloader
                            return args

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1
    if args.stage == args.partitions - 1:
        loss_file.close()


if __name__ == "__main__":
    now = time.time()
    args = main()
    if is_main_process():
        print("Total time taken {}".format(time.time() - now))
