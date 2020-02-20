# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning Bert for question-answering on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import signal

import torch.distributed as dist
from torch.multiprocessing import Process

import time
import datetime

import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np 
np.random.seed(42)
import random 
random.seed(42)

from torch.utils.data import (DataLoader, TensorDataset, DistributedSampler)
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)

from varuna import Varuna, load_varuna_checkpoint

from transformers import AdamW

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions)

TERMINATE_TRAINING = False
total_terminate_time = 0
blob_store_folder = "/home/core/myblobcontainer"

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer, stage_to_rank_map, train_state = None):

    if args.rank == 0:
        tb_writer = SummaryWriter()
    
    data_depth = len(stage_to_rank_map[0])
    filename = "train_reports/report-{}-{}-{}-{}_{}.csv".format(args.partitions, data_depth , args.per_gpu_train_batch_size, args.chunks, args.rank)
    of = open(filename, "w")

    of.write("MB time, TFLOPS, GPU mem, loss\n")

    epochs_done = minibatches_done = 0
    if train_state is not None:
        epochs_done = train_state["epoch"]
        minibatches_done = train_state["minibatches_done"]
    
    data_depth = len(stage_to_rank_map[0])
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    
    if args.report_name == "":
        args.report_name = "train_reports/report-{}-{}-{}-{}_{}.csv".format(args.partitions, data_depth , total_batch_size, args.chunks, args.rank)

    
    if os.path.exists(args.report_name) and args.resume:
        of = open(args.report_name, "a")
        of.write("morphed to {}-{}-{}\n".format(args.partitions, data_depth, total_batch_size))
    else:
        of = open(args.report_name, "w")
        of.write("MB time, TFLOPS, Max GPU mem, Curr GPU mem, loss\n")
    of.close()

    for stage in stage_to_rank_map:
        i = 0
        for rank in stage_to_rank_map[stage]:
            if rank == args.rank:
                rank_within_stage = i
                my_stage = stage
                break
            i += 1

    # Each stage divides data according to it's depth
    if len(stage_to_rank_map[my_stage]) > 1:
        num_replicas = len(stage_to_rank_map[my_stage])
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank_within_stage, shuffle=False)
        train_batch_size = args.train_batch_size // num_replicas
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, shuffle = False, drop_last = True)
    else:
        train_batch_size = args.train_batch_size
        train_sampler = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle = False, drop_last = True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.resume:
        optimizer.load_state_dict(torch.load(os.path.join(blob_store_folder,"opt_state.bin"),map_location="cpu"))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    dummy_inp = train_dataset[:1]
    inputs = {  'input_ids':       dummy_inp[0],
                'attention_mask':  dummy_inp[1], 
                'start_positions': dummy_inp[3], 
                'end_positions':   dummy_inp[4]}
    inputs['token_type_ids'] = dummy_inp[2]    

    model = Varuna(model, stage_to_rank_map, inputs, args.train_batch_size, optimizer, chunks=args.chunks, local_rank=args.local_rank)

    # Train!
    if args.rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per gpu = %d", train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = epochs_done * (len(train_dataloader) // args.gradient_accumulation_steps)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=my_stage!=(args.partitions-1), initial = epochs_done)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=my_stage!=(args.partitions-1))
        avg_mb_time = 0.0
        avg_tflops = 0.0

        for step, batch in enumerate(epoch_iterator):

            if global_step < minibatches_done:
                step += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}

            torch.cuda.reset_max_memory_allocated(args.device)
            
            minibatch_time = time.time()
            loss = model(inputs)
            minibatch_time = time.time() - minibatch_time
            tflops = 1.33 * (args.train_batch_size / minibatch_time)     # TFLOPS ~ 1.33 * examples per sec
            tflops = tflops / (args.partitions * data_depth)        # scale by # gpus
            avg_mb_time += minibatch_time
            avg_tflops += tflops

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                max_mem = torch.cuda.max_memory_allocated(args.device)
                curr_mem = torch.cuda.memory_allocated(args.device)
                of = open(args.report_name, "a")
                of.write("{}, {}, {}, {}, {}\n".format(minibatch_time, tflops, max_mem, curr_mem, loss))
                of.close()

                if my_stage == (args.partitions - 1) and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    print("loss at step",global_step, "is ", loss )
                    print("Current average pipeline time", avg_mb_time / (step+1) )
                    print("Avg TFLOPS", (avg_tflops / (step+1)) )
                    # logging_loss = tr_loss

            del batch, inputs

            if (args.max_steps > 0 and global_step > args.max_steps) or TERMINATE_TRAINING:
                start = time.time()
                model.checkpoint(blob_store_folder)
                print("Checkpoint time", time.time() - start)
                if args.rank == 0:
                    train_state = { "epoch": epoch, "minibatches_done": global_step }
                    torch.save(train_state, os.path.join(blob_store_folder,"train_state.bin") )
                    torch.save(optimizer.state_dict(), os.path.join(blob_store_folder,"opt_state.bin") )
                # all ranks must wait for each other before returning
                torch.distributed.barrier()
            
                epoch_iterator.close()
                break

        if (args.max_steps > 0 and global_step > args.max_steps) or TERMINATE_TRAINING :
            train_iterator.close()
            break

    if args.rank == 0:
        tb_writer.close()

    of.close()

    return global_step, tr_loss / global_step


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank != 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank == 0:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main(args, stage_to_rank_map):

    global total_terminate_time

    rank = args.rank
    size = args.partitions

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(True), args.fp16)

    # Set seed
    set_seed(args)

    for stage in stage_to_rank_map:
        if rank in stage_to_rank_map[stage]:
            my_stage = stage
            break
    print(args.rank, "with stage", my_stage)

    # Load pretrained model and tokenizer
    if args.resume:
        print("Resuming training!!")
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)        
        state_dict = load_varuna_checkpoint(my_stage, args.partitions, config.num_hidden_layers, blob_store_folder)
        model = BertForQuestionAnswering.from_pretrained(None, state_dict=state_dict, config=config)
        train_state = torch.load(os.path.join(blob_store_folder,"train_state.bin"), map_location="cpu" )
    else:
        if args.local_rank != 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        train_state = None 

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, stage_to_rank_map, train_state)

    total_terminate_time = time.time() - total_terminate_time
    print("Total terminate time", total_terminate_time)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Mini batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=150,       # myedits: default changed from 50
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,      # myedits: default changed from 50
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")


    parser.add_argument('--chunks', type=int, default=1, help="Number of micro-batches per mini-batch")
    parser.add_argument('--partitions', type=int, default=3, help='Number of devices over which the model is split')
    parser.add_argument('--stage_to_rank_map',type=str, default="",help="How GPU processes are divided among partitions")
    parser.add_argument('--rank', type=int, default=0, help='Partition index')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report_name', type=str, default="", help="file name for training report")
    args = parser.parse_args()

    def handler(signum,_):
        global TERMINATE_TRAINING, total_terminate_time
        print(args.rank, 'signal handler called with signal', signum)
        TERMINATE_TRAINING = True
        total_terminate_time = time.time()

    signal.signal(signal.SIGUSR1, handler)    

    connect_timeout = datetime.timedelta(minutes=4)
    dist.init_process_group('gloo', timeout=connect_timeout)

    # parse stage_to_rank_map
    stage_ranks = args.stage_to_rank_map.split(";", args.partitions)
    stage_to_rank_map = {}
    for i in range(args.partitions):
        ranks = stage_ranks[i].split(",")
        stage_to_rank_map[i] = [int(r) for r in ranks]

    main(args, stage_to_rank_map)