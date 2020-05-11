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

"""Pretrain GPT2"""

import torch

from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron import print_rank_0
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.model import GPT2Model
from megatron.training import pretrain, get_eval_numbers
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses

import signal

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=True)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if args.model_parallel_size > 1:
        data_b = mpu.broadcast_data(keys, data, datatype)  
    else:
        data_b = dict()
        for k in data:
            data_b[k] = data[k].cuda()

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def eval_step_varuna(data_iterator, model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    inputs = {tokens, position_ids, attention_mask, loss_mask, labels}
    inputs = dict({
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "labels": labels
    })
    output = model.evaluate(inputs)
    loss = output[0] if isinstance(output,list) else output

    # Reduce loss for logging.
    return loss, {'lm loss': loss}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()

    # Forward model.
    output = model(tokens, position_ids, attention_mask, loss_mask, labels)
    loss = output[0] if isinstance(output,list) else output

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}

def varuna_step(data_iterator, model):
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()

    # Pipeline model.
    inputs = dict({
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "labels": labels
    })

    loss = model(inputs)

    # Reduce loss for logging.
    # reduced_loss = reduce_losses([loss])
    return loss, {'lm loss': torch.Tensor([loss])}

def cond_forward_step(data_iterator, model):
    args = get_args()
    if args.varuna:
        return varuna_step(data_iterator, model)
    else:
        return forward_step(data_iterator, model)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    def handler(signum,_):
        print(torch.distributed.get_rank(), 'signal handler called with signal', signum)
        exit()

    signal.signal(signal.SIGUSR1, handler) 

    pretrain(train_valid_test_datasets_provider, model_provider, cond_forward_step,
             eval_step_varuna, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    # get_eval_numbers(train_valid_test_datasets_provider, model_provider,eval_step_varuna)

