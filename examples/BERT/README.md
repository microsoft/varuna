Varuna can be used with [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/)'s pytorch models. Here we've made the modifications for [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).

This example is built on [this commit](https://github.com/NVIDIA/DeepLearningExamples/commit/c481324031ecf0f70f8939516c02e16cac60446d) of the repo.
The diff file `bert.patch` in this example folder can be applied to this commit to get a working version of 
NVIDIA BERT with Varuna. 
~~~~
git clone https://github.com/NVIDIA/DeepLearningExamples/
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
git checkout c481324031ecf0f70f8939516c02e16cac60446d
cp /path/to/bert.patch ./
git apply bert.patch
~~~~
## Running

Please follow the instructions in the original repository along with instructions for Varuna training in the docs ("Launching Varuna" at docs/html/launching.html) for steps to run.

## Changes for Varuna Training

Some changes that are made in the patch are explained below.

Here, we mark the `CutPoint`s, the points in the model computation graph that are potential boundary points 
for the sequential stages in pipeline parallelism. They are marked by executing the `CutPoint` module at the desired points. Each `CutPoint` should be used only once in the forward computation. See docs for `CutPoint` for more details.
~~~~
diff --git a/PyTorch/LanguageModeling/BERT/modeling.py b/PyTorch/LanguageModeling/BERT/modeling.py
index 0a1bd8f..65ebecd 100755
--- a/PyTorch/LanguageModeling/BERT/modeling.py
+++ b/PyTorch/LanguageModeling/BERT/modeling.py
@@ -41,6 +41,8 @@ from torch.nn.parameter import Parameter
 import torch.nn.functional as F
 import torch.nn.init as init
 
+from varuna import CutPoint
+
 logger = logging.getLogger(__name__)
 
 PRETRAINED_MODEL_ARCHIVE_MAP = {

@@ -484,6 +486,7 @@ class BertEncoder(nn.Module):
     def __init__(self, config):
         super(BertEncoder, self).__init__()
         self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
+        self.cutpoints = nn.ModuleList([CutPoint() for _ in range(config.num_hidden_layers - 1)])
         self.output_all_encoded_layers = config.output_all_encoded_layers
         self._checkpoint_activations = False
 
@@ -518,6 +521,9 @@ class BertEncoder(nn.Module):
 
                 if self.output_all_encoded_layers:
                     all_encoder_layers.append(hidden_states)
+                
+                if i < len(self.cutpoints):
+                    hidden_states = self.cutpoints[i](hidden_states)
 
         if not self.output_all_encoded_layers or self._checkpoint_activations:
             all_encoder_layers.append(hidden_states)
~~~~

Shared parameters (parameters that are used across `CutPoint`s) such as embedding 
weight and prediction head weight in transformer models should be marked as two separate parameters.
These parameters should then be passed to the `Varuna` instance by their names (see `Varuna` class docs).
This allows varuna to partition across these cutpoints and sync these parameters while updating them.
Here, `decoder.weight` is a new tensor parameter copied from the word embedding weight.

~~~~
diff --git a/PyTorch/LanguageModeling/BERT/modeling.py b/PyTorch/LanguageModeling/BERT/modeling.py
index 0a1bd8f..65ebecd 100755
--- a/PyTorch/LanguageModeling/BERT/modeling.py
+++ b/PyTorch/LanguageModeling/BERT/modeling.py
@@ -558,7 +564,7 @@ class BertLMPredictionHead(nn.Module):
         self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                  bert_model_embedding_weights.size(0),
                                  bias=False)
-        self.decoder.weight = bert_model_embedding_weights
+        self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
         self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
 
     def forward(self, hidden_states):
~~~~

For varuna training, the model passed should include complete computation with the loss function.
So, creating a new combined model here.
~~~~
diff --git a/PyTorch/LanguageModeling/BERT/run_pretraining.py b/PyTorch/LanguageModeling/BERT/run_pretraining.py
index a357788..cb103b0 100755
--- a/PyTorch/LanguageModeling/BERT/run_pretraining.py
+++ b/PyTorch/LanguageModeling/BERT/run_pretraining.py
@@ -131,6 +132,16 @@ class BertPretrainingCriterion(torch.nn.Module):
         total_loss = masked_lm_loss + next_sentence_loss
         return total_loss
 
+class BertForPreTrainingWithCriterion(torch.nn.Module):
+    def __init__(self, config):
+        super(BertForPreTrainingWithCriterion, self).__init__()
+        self.model = modeling.BertForPreTraining(config)
+        self.criterion = BertPretrainingCriterion(config.vocab_size)
+    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_labels):
+        prediction_scores, seq_relationship_score = self.model(input_ids, token_type_ids, attention_mask)
+        loss = self.criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
+        return loss
+
 
 def parse_arguments():
~~~~

Arguments to enable varuna training. Other than flags '--varuna' and '--profiling', 
these are passed by the varuna launcher. These are later used while initializing the `Varuna` instance.
See "Launching Varuna" in docs for more details.

~~~~ 
@@ -278,6 +289,17 @@ def parse_arguments():
                         help='Disable tqdm progress bar')
     parser.add_argument('--steps_this_run', type=int, default=-1,
                         help='If provided, only run this many steps before exiting')
+    parser.add_argument('--varuna', action='store_true',
+                        help='Flag to enable training with Varuna')
+    parser.add_argument('--stage_to_rank_map', default=None, type=str,
+                        help = "stage to rank map of Varuna model")
+    parser.add_argument("--chunk_size", type=int,default=None,
+                        help = "number of microbatches for pipeline")
+    parser.add_argument("--batch-size", type=int, default=None,
+                        help = "per-process batch size given by varuna")
+    parser.add_argument("--rank", type=int, default=-1)
+    parser.add_argument("--profiling", action='store_true', 
+                        help="whether to run profiling for Varuna")
 
     args = parser.parse_args()
     args.fp16 = args.fp16 or args.amp
~~~~

Varuna distributed training requires a GLOO backend for pipeline parallelism.
Varuna handles pipeline & data parallelism, micro-batches (gradient accumulation) 
and mixed precision training internally, so allreduce and accumulation need not be handled by the user.
It automatically splits a fixed global batch size over the available GPUs and the launcher specifies the 
per GPU batch size as the `batch-size` argument. This is used for varuna instead of original `train_batch_size`.
~~~~
@@ -300,10 +322,10 @@ def setup_training(args):
         torch.cuda.set_device(args.local_rank)
         device = torch.device("cuda", args.local_rank)
         # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
-        torch.distributed.init_process_group(backend='nccl', init_method='env://')
+        torch.distributed.init_process_group(backend='gloo' if args.varuna else 'nccl', init_method='env://')
         args.n_gpu = 1
         
-    if args.gradient_accumulation_steps == 1:
+    if args.varuna or (args.gradient_accumulation_steps == 1):
         args.allreduce_post_accumulation = False
         args.allreduce_post_accumulation_fp16 = False
         
@@ -317,6 +339,9 @@ def setup_training(args):
     print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
         device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
 
+    if args.varuna:
+        args.gradient_accumulation_steps = 1
+        args.train_batch_size = args.batch_size
     if args.gradient_accumulation_steps < 1:
         raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
             args.gradient_accumulation_steps))
~~~~

The `nn.Module` for training needs to be wrapped in a `Varuna` instance with all necessary parameters.
The `stage_to_rank_map` passed the varuna launcher can be used to get the parallel configuration.
For profiling and automatic partitioning, varuna requires a few sample inputs, for which a `get_batch_fn`
function needs to be passed. This should return a sample batch of given size as a dictionary.
~~~~
@@ -338,7 +363,7 @@ def setup_training(args):
 
     return device, args
 
-def prepare_model_and_optimizer(args, device):
+def prepare_model_and_optimizer(args, device, get_dict_batch):
 
     # Prepare model
     config = modeling.BertConfig.from_json_file(args.config_file)
@@ -348,9 +373,27 @@ def prepare_model_and_optimizer(args, device):
         config.vocab_size += 8 - (config.vocab_size % 8)
 
     modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
-    model = modeling.BertForPreTraining(config)
+    model = BertForPreTrainingWithCriterion(config)
+
+    if args.varuna:
+        data_file = None
+        for f in os.listdir(args.input_dir):
+            if os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f:
+                data_file = os.path.join(args.input_dir, f)
+                break
+        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
+        def get_batch_fn(size, device=None):
+            batch = next(iter(torch.utils.data.DataLoader(train_data, batch_size=size)))
+            return get_dict_batch(batch, device=device)
+        # shared_weights = []
+        pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
+        global_batch_size = args.train_batch_size * data_parallel_size
+        shared_weights = [("model.bert.embeddings.word_embeddings.weight", "model.cls.predictions.decoder.weight")]
+        model = Varuna(model, args.stage_to_rank_map, get_batch_fn, global_batch_size, 
+                args.chunk_size, fp16=args.fp16, device=device, shared_weights=shared_weights)
~~~~

For varuna, the model and optimizer state are checkpointed and loaded together and need not be 
handled separately. The checkpoint write and read functions must be called on all ranks of the process.
Detailed options for checkpointing are described in docs.

~~~~
@@ -365,7 +408,10 @@ def prepare_model_and_optimizer(args, device):
         else:
             checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
 
-        model.load_state_dict(checkpoint['model'], strict=False)
+        # varuna ckpt will be loaded later with optimizer
+        if not args.varuna:
+            model.load_state_dict(checkpoint['model'], strict=False)
+        load_iter = global_step
         
         if args.phase2 and not args.init_checkpoint:
             global_step -= args.phase1_end_step
@@ -385,7 +431,9 @@ def prepare_model_and_optimizer(args, device):
     lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                        warmup=args.warmup_proportion, 
                                        total_steps=args.max_steps)
-    if args.fp16:
+    if args.varuna:
+        model.set_optimizer(optimizer, loss_scale = "dynamic", init_loss_scale = args.init_loss_scale)
+    elif args.fp16:
 
         if args.loss_scale == 0:
             model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", cast_model_outputs=torch.float16)
@@ -393,30 +441,34 @@ def prepare_model_and_optimizer(args, device):
             model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale, cast_model_outputs=torch.float16)
         amp._amp_state.loss_scalers[0]._loss_scale = args.init_loss_scale
 
-    model.checkpoint_activations(args.checkpoint_activations)
+    if not args.varuna:
+        model.checkpoint_activations(args.checkpoint_activations)
 
     if args.resume_from_checkpoint:
+        if not args.varuna:
+            optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)
+        else:
+            model.load_checkpoint(args.output_dir, load_iter)
         if args.phase2 or args.init_checkpoint:
~~~~

When using `Varuna`, the user does not need to wrap in fp16 modules or `DistributedDataParallel`.
`Varuna` internally handles all parallelisation, mixed-precision and transfers the model to the given GPU
for each worker. The module passed to Varuna should therefore be an unmodified instance of the model on the CPU. 
~~~~ 
         # Restore AMP master parameters          
-        if args.fp16:
+        if args.fp16 and not args.varuna:
             optimizer._lazy_init_maybe_master_weights()
             optimizer._amp_stash.lazy_init_called = True
             optimizer.load_state_dict(checkpoint['optimizer'])
             for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                 param.data.copy_(saved_param.data)
 
-    if args.local_rank != -1:
+    if args.local_rank != -1 and not args.varuna:
         if not args.allreduce_post_accumulation:
             model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
         else:

~~~~

Varuna `step` function requires inputs as dictionaries.
~~~~
@@ -503,7 +556,19 @@ def main():
     dllogger.log(step="PARAMETER", data={"Config": [str(args)]})
 
     # Prepare optimizer
-    model, optimizer, lr_scheduler, checkpoint, global_step, criterion = prepare_model_and_optimizer(args, device)
+    def get_dict_batch(batch, device=None):
+        if device is not None:
+            batch = [t.to(device) for t in batch]
+        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
+        batch = dict({"input_ids": input_ids, 
+                    "token_type_ids": segment_ids, 
+                    "attention_mask":input_mask, 
+                    "masked_lm_labels": masked_lm_labels, 
+                    "next_sentence_labels": next_sentence_labels})
+        return batch
+        
+
+    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device, get_dict_batch)
 
     if is_main_process():
         dllogger.log(step="PARAMETER", data={"SEED": args.seed})
~~~~

The varuna training loop has forward and backward combined into a single `step` call that takes dictionary inputs, handles gradient accumulation and reduce and returns a boolean for fp16 overflow. (See docs for `Varuna` class)
~~~~
@@ -588,24 +660,27 @@ def main():
                 for step, batch in enumerate(train_iter):
 
                     training_steps += 1
-                    batch = [t.to(device) for t in batch]
-                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
-                    prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
-                    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
-                    if args.n_gpu > 1:
-                        loss = loss.mean()  # mean() to average on multi-gpu.
-
-                    divisor = args.gradient_accumulation_steps
-                    if args.gradient_accumulation_steps > 1:
-                        if not args.allreduce_post_accumulation:
-                            # this division was merged into predivision
-                            loss = loss / args.gradient_accumulation_steps
-                            divisor = 1.0
-                    if args.fp16:
-                        with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
-                            scaled_loss.backward()
+                    batch = get_dict_batch(batch, device=device)
+                    if args.varuna:
+                        loss, overflow, grad_norm = model.step(batch)
+                        loss = torch.tensor(loss)
+                        divisor = 1
                     else:
-                        loss.backward()
+                        loss = model(**batch)
+                        if args.n_gpu > 1:
+                            loss = loss.mean()  # mean() to average on multi-gpu.
+
+                        divisor = args.gradient_accumulation_steps
+                        if args.gradient_accumulation_steps > 1:
+                            if not args.allreduce_post_accumulation:
+                                # this division was merged into predivision
+                                loss = loss / args.gradient_accumulation_steps
+                                divisor = 1.0
+                        if args.fp16:
+                            with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
+                                scaled_loss.backward()
+                        else:
+                            loss.backward()
                     average_loss += loss.item()
 
                     if training_steps % args.gradient_accumulation_steps == 0:
~~~~
