CutPoints
==============

Varuna slices a DNN model into sequential stages for pipeline parallelism. 
For this, the model should be annotated with varuna :class:`CutPoint` instances between 
different operations/ parts of model computation.

A :class:`CutPoint` in varuna is an abstraction to mark a potential point of partitioning in your model.
It is implemented as a :class:`torch.nn.Module` instance, which is called on the activations at the potential
boundary point. For each :class:`CutPoint`, Varuna can either ignore it or activate it as a partition boundary. 
CutPoints can be marked anywhere in the model as follows:

.. code-block:: python

    from varuna import CutPoint

    class SampleModel(nn.Module):
    def __init__(...):
        ....
        self.cutpoints = [CutPoint() for i in range(num_cutpoints)]
        ....

    def forward(input...):
        input = self.some_operation(input)
        input = self.cutpoints[0](input)     # marked as a potential stage boundary
        input = self.some_other_operation(input)
        ....
        for i in range(sub_modules):
        x = sub_module_i(input, ...)
        x = self.cutpoints[i+1](x)        # each cutpoint instance should be used only once in a model
        ....


Based on the number of desired pipeline stages, Varuna chooses a subset of the given cutpoints and 
activates them as actual boundaries between stages. For example, if the user marks `n` cutpoints in total, 
and wants 4 parallel pipeline stages, 3 cutpoints will be activated as partitions between the 4 stages and the rest 
`n-3` are treated as they don't exist.
|br| With this partitioning, each worker in the distributed job runs a sub-section of the model code between
two activated :class:`CutPoint` instances, or between one activated :class:`CutPoint` and the
beginning/end of the model.

For an activated :class:`CutPoint`, the input to the cutpoint is an intermediate activation in the model
that needs to be passed between sequential stages.

.. note::

    The input to any :class:`CutPoint` in the model's execution should be a single :class:`torch.Tensor`
    of shape `(b, d2, d3, ...)` where `b` is the number of examples in the input to the model.
    |br| This is important because Varuna uses micro-batches to parallelize computation and relies on this
    format for communication between pipeline stages.

Operations separated by CutPoints should preferably have no shared modules/parameters. 
For weight sharing between different parts of the module, you should register separate :class:`nn.Parameter` 
instances (even for the same tensor) and pass the pair of parameter names as :attr:`shared_weights` to the
:class:`Varuna` object.

For example, in language models like BERT and GPT2, the weights for word embedding computation at
the beginning of the model are also utilised at the end of the model for prediction logits. 
So, if this weight is wrapped in two separate :class:`torch.nn.Parameter` instances, they will have two 
corresponding "parameter names" (string values) in the model (see :func:`named_parameters` for :class:`torch.nn.Parameter`).
These can be passed as a pair of names for each shared weight to :class:`Varuna` as follows:

.. code-block:: python

    # list of 2-tuples with parameter names
    shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]   
    model = Varuna( model, args.stage_to_rank_map, dry_run_input, global_batch_size, 
                        args.chunk_size, args.fp16,
                        local_rank=args.local_rank, 
                        device=args.local_rank, 
                        shared_weights=shared_weights)  # passed to varuna init


.. |br| raw:: html

    <br/>