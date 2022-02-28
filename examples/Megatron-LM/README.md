Varuna can be used with [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)'s pytorch models. Here we've made required modifications for GPT-2 training.

This example is built on [this commit](https://github.com/NVIDIA/Megatron-LM/commit/28cd66e1a2d01ada962c356f4a23961cbf4c00b3) of the Megatron-LM repo. An older commit is given as an
example here because the versions after this implement custom pipeline parallelism.
The diff file `megatron.patch` in this example folder can be applied to this commit to get a working version of 
Megatron-LM GPT-2 with Varuna. 
~~~~
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM
git checkout 28cd66e1a2d01ada962c356f4a23961cbf4c00b3
cp /path/to/megatron.patch ./
git apply megatron.patch
~~~~
The scripts in this folder can then be used for profiling (profile_gpt2.sh) and pretraining (pretrain_gpt2_varuna.sh) with varuna.

## Running

Pretraining for the Megatron-LM model can be triggered using the `pretrain_gpt2_varuna.sh` in this folder, once the code is set up.
Parameters for different sized models (355M, 1.5B, 2.5B, 8.3B) are given in the script and can be commented/un-commented as required.
Parameters for the dataset and model write path will need to be set before running. The no
Varuna parallelisation requires GPUs. FP16 training requires specialized V100 GPUs.
~~~~
bash pretrain_gpt2_varuna.sh
~~~~
Please refer to the "Launching Varuna" page in the docs (available at docs/html/launching.html) for more details on how to run.



## Changes for Varuna Training

Some of the changes in the Megatron patch are explained below.


Arguments to enable varuna training. Other than flags '--varuna' and '--profiling', 
these are passed by the varuna launcher. These are later used while initializing the `Varuna` instance.

~~~~

diff --git a/megatron/arguments.py b/megatron/arguments.py
@@@ -300,6 +302,20 @@ def _add_learning_rate_args(parser):
 
     return parser
 
+def _add_varuna_args(parser):
+    group = parser.add_argument_group(title='varuna')
+
+    group.add_argument("--varuna", action='store_true', default=False,
+                        help = "Enable varuna pipeline training")
+    group.add_argument("--stage_to_rank_map", type=str, default=None,
+                        help = "stage to rank map of Varuna model")
+    group.add_argument("--chunk_size", type=int,default=None,
+                        help = "number of microbatches for pipeline")
+    group.add_argument("--rank", type=int, default=-1,
+                        help = "global rank passed by varuna launcher")
+    group.add_argument("--resume_step", type=int, default=None,
+                        help = "Iteration to resume training, given by varuna morphing")
+    group.add_argument("--profiling", action='store_true', 
+                        help="whether to run profiling for Varuna")
+    return parser
 
 def _add_checkpointing_args(parser):
     group = parser.add_argument_group(title='checkpointing')

~~~~

For varuna, the model and optimizer state are checkpointed and loaded together and need not be 
handled separately. The checkpoint write and read functions must be called on all ranks of the process.
Detailed options for checkpointing are described in docs.

~~~~

diff --git a/megatron/checkpointing.py b/megatron/checkpointing.py
@@ -96,18 +96,19 @@ def save_checkpoint(iteration, model, optimizer, lr_scheduler):
     # Only rank zero of the data parallel writes to the disk.
     if isinstance(model, torchDDP):
         model = model.module
-    if mpu.get_data_parallel_rank() == 0:
+    if args.rank==0 or (not args.varuna and mpu.get_data_parallel_rank() == 0):
 
         # Arguments, iteration, and model.
         state_dict = {}
         state_dict['args'] = args
         state_dict['checkpoint_version'] = 2.0
         state_dict['iteration'] = iteration
-        state_dict['model'] = model.state_dict_for_save_checkpoint()
+        if not args.varuna:
+            state_dict['model'] = model.state_dict_for_save_checkpoint()
 
         # Optimizer stuff.
         if not args.no_save_optim:
-            if optimizer is not None:
+            if optimizer is not None and not args.varuna:
                 state_dict['optimizer'] = optimizer.state_dict()
             if lr_scheduler is not None:
                 state_dict['lr_scheduler'] = lr_scheduler.state_dict()
@@ -129,10 +130,14 @@ def save_checkpoint(iteration, model, optimizer, lr_scheduler):
         torch.save(state_dict, checkpoint_name)
         print('  successfully saved {}'.format(checkpoint_name))
 
+    ckpt_future = None
+    if args.varuna:
+        ckpt_future = model.checkpoint(args.save, tempdir=None, step=iteration)
+
     # Wait so everyone is done (necessary)
     torch.distributed.barrier()
     # And update the latest iteration
-    if torch.distributed.get_rank() == 0:
+    if ckpt_future is not None and torch.distributed.get_rank() == 0:
         tracker_filename = get_checkpoint_tracker_filename(args.save)
         with open(tracker_filename, 'w') as f:
             f.write(str(iteration))
 
@@ -223,12 +233,15 @@ def load_checkpoint(model, optimizer, lr_scheduler, load_arg='load'):
         print_rank_0('could not find arguments in the checkpoint ...')
 
     # Model.
-    model.load_state_dict(state_dict['model'])
+    if not args.varuna:
+        model.load_state_dict(state_dict['model'])
+    else:
+        model.load_checkpoint(args.load, iteration)
 
     # Optimizer.
     if not release and not args.finetune and not args.no_load_optim:
         try:
-            if optimizer is not None:
+            if optimizer is not None and not args.varuna:
                 optimizer.load_state_dict(state_dict['optimizer'])
             if lr_scheduler is not None:
                 lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

~~~~


Shared parameters (parameters that are used across `CutPoint`s) such as embedding 
weight and prediction head weight in transformer models should be marked as two separate parameters.
These parameters should then be passed to the `Varuna` instance by their names (see `Varuna` class docs).
This allows varuna to partition across these cutpoints and sync these parameters while updating them.
Here, `lm_head_weight` is a new tensor parameter copied from the original embedding weight.

Another change in this file is to include the loss mask operation within the `nn.Module` so that the `Varuna`
engine can compute the loss and run backward with the passed module.

~~~~

diff --git a/megatron/model/gpt2_model.py b/megatron/model/gpt2_model.py
index b0d275f..e4b7dba 100644
--- a/megatron/model/gpt2_model.py
+++ b/megatron/model/gpt2_model.py
@@ -50,9 +50,14 @@ class GPT2Model(MegatronModule):
             scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                          args.num_layers))
 
+        if args.varuna:
+            self.lm_head_weight = torch.nn.Parameter(self.language_model.embedding.word_embeddings.weight)
+        else:
+            self.lm_head_weight = self.language_model.embedding.word_embeddings.weight
+
     def forward(self, input_ids, position_ids, attention_mask, labels=None,
                 tokentype_ids=None, layer_past=None, get_key_value=False,
-                forward_method_parallel_output=None):
+                forward_method_parallel_output=None, loss_mask=None):
 
         # Language model.
         lm_output = self.language_model(input_ids,
@@ -71,7 +76,7 @@ class GPT2Model(MegatronModule):
             parallel_output = forward_method_parallel_output
         output = parallel_lm_logits(
             lm_output,
-            self.language_model.embedding.word_embeddings.weight,
+            self.lm_head_weight,
             parallel_output)
 
         if get_key_value:
@@ -85,6 +90,11 @@ class GPT2Model(MegatronModule):
                 loss = mpu.vocab_parallel_cross_entropy(output, labels)
             else:
                 loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
+            
+            if loss_mask is not None:
+                loss_mask = loss_mask.view(-1)
+                loss = torch.sum(loss.view(-1) * loss_mask) / loss_mask.sum()
+            
             return loss
 
 
~~~~

Here, we mark the `CutPoint`s, the points in the model computation graph that are potential boundary points 
for the sequential stages in pipeline parallelism. They are marked by executing the `CutPoint` module at the desired points. Each `CutPoint` should be used only once in the forward computation.

~~~~

diff --git a/megatron/model/transformer.py b/megatron/model/transformer.py
index f2be536..2efe5f2 100644
--- a/megatron/model/transformer.py
+++ b/megatron/model/transformer.py
@@ -520,6 +522,7 @@ class ParallelTransformer(MegatronModule):
                 output_layer_init_method, layer_number)
         self.layers = torch.nn.ModuleList(
             [build_layer(i + 1) for i in range(self.num_unique_layers)])
+        self.cutpoints = torch.nn.ModuleList([CutPoint() for i in range(self.num_layers-1)])
 
         # Print layer ordering.
         if self.num_layers != self.num_unique_layers:
@@ -581,7 +584,8 @@ class ParallelTransformer(MegatronModule):
                 'activation checkpointing'
 
         # data format change to avoid explicit tranposes : [b s h] --> [s b h]
-        hidden_states = hidden_states.transpose(0, 1).contiguous()
+        if hidden_states is not None and hidden_states.dim() >= 2:
+            hidden_states = hidden_states.transpose(0, 1).contiguous()
 
         if self.checkpoint_activations:
             hidden_states = self._checkpointed_forward(hidden_states,
@@ -598,6 +602,8 @@ class ParallelTransformer(MegatronModule):
                                       attention_mask,
                                       layer_past=past,
                                       get_key_value=get_key_value)
+                if index < (self.num_layers-1):
+                    hidden_states = self.cutpoints[index](hidden_states)
                 if get_key_value:
                     hidden_states, present = hidden_states
                     presents.append(present)

~~~~

For `Varuna`, a `get_batch` function needs to be defined to get sample inputs to the
model with given batch sizes. This is used for one-time profiling of the model before training 
and needs to be passed to the `Varuna` instance.

~~~~
diff --git a/megatron/training.py b/megatron/training.py
index ca1bd26..d2150ca 100644
--- a/megatron/training.py
+++ b/megatron/training.py
 @@ -74,32 +83,42 @@ def pretrain(train_valid_test_dataset_provider, model_provider,
     args = get_args()
     timers = get_timers()
 
+    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(train_valid_test_dataset_provider)
+    def get_batch_fn(size, device=None):
+        sample_iter = iter(torch.utils.data.DataLoader(train_ds, batch_size=size))
+        return get_batch(sample_iter, device=device) 
+
     # Model, optimizer, and learning rate.
     timers('model and optimizer').start()
-    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
+    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, get_batch_fn)
     timers('model and optimizer').stop()
 
~~~~

Calling profiling for varuna for different micro-batch sizes. See `Profiler` docs.

~~~~
     timers('train/valid/test data iterators').stop()
 
     # Print setup timing.
     print_rank_0('done with setups ...')
     timers.log(['model and optimizer', 'train/valid/test data iterators'])
+    
+    if args.varuna and args.profiling:
+        profile = model.profile_all(list(range(1,25)))
+        return
+    
     print_rank_0('training ...')
 
     iteration = 0
~~~~
 
The `nn.Module` for training needs to be wrapped in a `Varuna` instance with all necessary parameters.
The `stage_to_rank_map` passed the varuna launcher can be used to get the parallel configuration.
When using `Varuna`, the user does not need to wrap in fp16 modules or `DistributedDataParallel`.
`Varuna` internally handles all parallelisation, mixed-precision and transfers the model to the given GPU
for each worker. The module passed to Varuna should therefore be an unmodified instance of the model on the CPU. 

~~~~
-def get_model(model_provider_func):
+def get_model(model_provider_func, get_batch_fn=None):
     """Build the model."""
     args = get_args()
 
     # Build model on cpu.
     model = model_provider_func()
+    profiler = None
+    if args.varuna:
+        assert get_batch_fn is not None, "Must provide get_batch_fn to varuna"
+        shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]
+
+        if args.profiling:
+            profiler = Profiler(model, get_batch_fn, fp16=args.fp16, device = args.local_rank,
+                        from_cache=True, out_folder=args.save, add_to_existing=True)
+        else:
+            pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
+            args.partitions = pipeline_parallel_size
+            global_batch_size = args.batch_size * data_parallel_size
+            model = Varuna( model, args.stage_to_rank_map, get_batch_fn, global_batch_size, 
+                            args.chunk_size, args.fp16, local_rank=args.local_rank, 
+                            device=args.local_rank, shared_weights=shared_weights)
 
     # Print number of parameters.
     if mpu.get_data_parallel_rank() == 0:
@@ -127,26 +161,29 @@ def get_model(model_provider_func):
             mpu.get_model_parallel_rank(),
             sum([p.nelement() for p in model.parameters()])), flush=True)
 
-    # GPU allocation.
-    model.cuda(torch.cuda.current_device())
+    # Varuna handles fp16, parallelisation, moving to devices internally
+    if not args.varuna:
+        # GPU allocation.
+        model.cuda(torch.cuda.current_device())
 
-    # Fp16 conversion.
-    if args.fp16:
-        model = FP16_Module(model)
-
-    # Wrap model for distributed training."""
-    if args.DDP_impl == 'torch':
-        i = torch.cuda.current_device()
-        model = torchDDP(model, device_ids=[i], output_device=i,
-                         process_group=mpu.get_data_parallel_group())
-        return model
-    if args.DDP_impl == 'local':
-        model = LocalDDP(model)
-        return model
+        # Fp16 conversion.
+        if args.fp16:
+            model = FP16_Module(model)
+
+        # Wrap model for distributed training."""
+        if args.DDP_impl == 'torch':
+            i = torch.cuda.current_device()
+            model = torchDDP(model, device_ids=[i], output_device=i,
+                            process_group=mpu.get_data_parallel_group())
+            return model
+        if args.DDP_impl == 'local':
+            model = LocalDDP(model)
+            return model
 
-    raise NotImplementedError('Unknown DDP implementation specified: {}. '
-                              'Exiting.'.format(args.DDP_impl))
+        raise NotImplementedError('Unknown DDP implementation specified: {}. '
+                                'Exiting.'.format(args.DDP_impl))
 
+    return model, profiler
 
 def get_optimizer(model):
     """Set up the optimizer."""
~~~~

The `set_optimizer` function must be called on the `Varuna` instance for training.
~~~~
@@ -206,12 +243,16 @@ def get_learning_rate_scheduler(optimizer):
     return lr_scheduler
 
 
-def setup_model_and_optimizer(model_provider_func):
+def setup_model_and_optimizer(model_provider_func, get_batch_fn):
     """Setup model and optimizer."""
     args = get_args()
 
-    model = get_model(model_provider_func)
+    model, profiler = get_model(model_provider_func, get_batch_fn)
     optimizer = get_optimizer(model)
+    if args.varuna:
+        if args.profiling:
+            model = profiler
+        model.set_optimizer(optimizer, init_loss_scale=2**17, min_loss_scale=args.min_scale)
     lr_scheduler = get_learning_rate_scheduler(optimizer)
 
     if args.load is not None:
~~~~

For varuna, the training loop only involves a single `step` function instead of forward and backward passes.

~~~~
 
@@ -269,34 +310,47 @@ def backward_step(optimizer, model, loss):
     timers('backward-clip-grad').stop()
 
 
-def train_step(forward_step_func, data_iterator,
-               model, optimizer, lr_scheduler):
+def train_step(forward_or_varuna_step_func, data_iterator,
+               model, optimizer, lr_scheduler, iteration):
     """Single training step."""
     args = get_args()
     timers = get_timers()
 
-    # Forward model for one step.
-    timers('forward').start()
-    loss, loss_reduced = forward_step_func(data_iterator, model)
-    timers('forward').stop()
-
-    # Calculate gradients, reduce across processes, and clip.
-    timers('backward').start()
-    backward_step(optimizer, model, loss)
-    timers('backward').stop()
-
+    if not args.varuna:
+        forward_step_func = forward_or_varuna_step_func
+        # Forward model for one step.
+        timers('forward').start()
+        loss, loss_reduced = forward_step_func(data_iterator, model)
+        timers('forward').stop()
+
+        # Calculate gradients, reduce across processes, and clip.
+        timers('backward').start()
+        backward_step(optimizer, model, loss)
+        timers('backward').stop()
+        overflow = False
+        global_norm = -1
+    else:
+        # timers('varuna')
+        varuna_step_func = forward_or_varuna_step_func
+        loss, loss_reduced, overflow, global_norm = \
+                varuna_step_func(data_iterator, model)
+  
     # Update parameters.
     timers('optimizer').start()
-    optimizer.step()
+    if not overflow:  
+        optimizer.step()
+    model.zero_grad()
+    
+    overflow = optimizer.overflow if (args.fp16 and not args.varuna) else overflow
     timers('optimizer').stop()
 
     # Update learning rate.
     skipped_iter = 0
-    if not (args.fp16 and optimizer.overflow):
+    if not overflow:
         lr_scheduler.step()
     else:
         skipped_iter = 1
-
+        
     return loss_reduced, skipped_iter
 
~~~~

Varuna requires the input batches to be sharded accross data parallel ranks. Different pipeline parallel ranks with
the same data parallel ranks will have the same input batch of batch size `global_batch_size / data_parallel_size`,
and all data parallel ranks together will make up the effective batch processed by the model. 
~~~~
diff --git a/megatron/utils.py b/megatron/utils.py
index 24d832d..556770b 100644
--- a/megatron/utils.py
+++ b/megatron/utils.py
@@ -96,8 +97,13 @@ def make_data_loader(dataset):
     args = get_args()
 
     # Data parallel arguments.
-    world_size = mpu.get_data_parallel_world_size()
-    rank = mpu.get_data_parallel_rank()
+    if args.varuna:
+        pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
+        pipeline_stage, data_parallel_rank = get_this_rank_config_varuna(args.stage_to_rank_map, args.rank)
+        world_size = data_parallel_size; rank = data_parallel_rank
+    else:
+        world_size = mpu.get_data_parallel_world_size()
+        rank = mpu.get_data_parallel_rank()
     global_batch_size = args.batch_size * world_size
     num_workers = args.num_workers
~~~~

Input to `Varuna` should be in the form of a python dictionary fo argument names to values.
The loss returned by `Varuna` is a float synced across all workers and the overflow boolean indicates 
fp16 overflow (if applicable).
~~~~
 
diff --git a/pretrain_gpt2.py b/pretrain_gpt2.py
index 372258f..0cf832d 100644
--- a/pretrain_gpt2.py
+++ b/pretrain_gpt2.py
@@ -66,6 +68,17 @@ def get_batch(data_iterator):
         args.reset_attention_mask,
         args.eod_mask_loss)
 
+
+    if args.varuna:
+        inputs = dict({
+            "input_ids": tokens,
+            "position_ids": position_ids,
+            "attention_mask": attention_mask,
+            "loss_mask": loss_mask,
+            "labels": labels
+        })
+        return inputs
+    
     return tokens, labels, loss_mask, attention_mask, position_ids
 
 
@@ -89,6 +102,41 @@ def forward_step(data_iterator, model):
 
     return loss, {'lm loss': reduced_loss[0]}
 
+def varuna_step(data_iterator, model):
+
+    args = get_args()
+    timers = get_timers()
+
+    # Get the batch.
+    timers('batch generator').start()
+    inputs = get_batch(data_iterator)
+    timers('batch generator').stop()
+
+    loss, overflow, global_norm = model.step(inputs)
+    loss = torch.Tensor([loss]).cuda()
+
+    return loss, {'lm loss': loss}, overflow, global_norm
~~~~
