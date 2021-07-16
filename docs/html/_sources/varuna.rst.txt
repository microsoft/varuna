The Varuna class
================

.. py:currentmodule:: varuna

The torch.nn.Module object for your DNN model should be wrapped in a :class:`Varuna` instance for training.
This class extends torch.nn.Module and handles distributed pipeline & data parallelism, mixed precision and 
shared parameter weights internally.

Wrapping in :class:`Varuna` partitions the model into pipeline stages across the distributed job.
For this, it uses stage allocation information that is passed by ``varuna.launcher`` to all
worker processes. The launcher uses a string argument ``stage_to_rank_map`` which must be parsed
and used for :class:`Varuna` initialisation. (see :doc:`launching`)

For profiling and automatic partitioning, Varuna needs sample inputs.
For this, a ``get_batch_fn`` needs to be passed during initialisation which returns a sample input batch
of a given size. This is used to profile the model's computation graph and should return
a dictionary of keywords to args, similar to the ``step`` function.

The model passed to :class:`Varuna` should be on CPU. Once the profiling and partitioning are done,
the model is moved to the assigned GPU. So the user need not do ``model.cuda()`` anywhere.

Optimizer creation should be after wrapping in :class:`Varuna`, since it requires model parameters as input.
The optimizer needs to be registered with Varuna using a setter.

Example:

.. code-block:: python

   model = MyModel()             # full model on CPU
   def get_batch_fn(size, device=None):
      batch = dataset[:size]
      if device is not None:
         batch = [t.to(device) for t in batch]
      inputs, mask = batch
      return {'inputs': inputs, 'mask': mask, 'extra_norm': True }
   # parameter sharing across the model, marked as pairs of param_names
   shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]  
   model = Varuna( model, args.stage_to_rank_map, get_batch_fn, global_batch_size, 
                     args.chunk_size, args.fp16, local_rank=args.local_rank, 
                     device=args.local_rank, shared_weights=shared_weights)

   # now model is a subset of the original model, moved to the GPU on each process

   optimizer = get_optimizer(model)
   model.set_optimizer(optimizer)

.. autoclass:: Varuna
   
   .. automethod:: set_optimizer
   .. automethod:: step
   .. automethod:: checkpoint
   .. automethod:: load_checkpoint
   .. automethod:: evaluate
