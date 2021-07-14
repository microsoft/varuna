Launching Varuna
================

Varuna distributed training is run as a set of processes for each GPU on each machine. It uses PyTorch's 
distributed framework and must be used with the gloo backend.
This distributed process is triggered from a single machine with a list of reachable machines (IPs) as 
`machine_list` and `gpus_per_node` GPUs on each node. This triggering machine is usually the 'manager' 
(as explained in :doc:`morphing`). Morphing is enabled by default, to disable it use the `no_morphing` flag.
Training with varuna can be run with the `run_varuna` module as follows:

.. code-block:: bash

    python -m varuna.run_varuna --machine_list <file_with_ips> --gpus_per_node <num_gpus_per_node> 
    --batch-size <total_effective_batch_size> --nstages <number_of_pipeline_stages> 
    --chunk_size <micro_batch_size_for_pipeline> 
    --code_dir <working_dir_for_training> user_training_script.py <...user args...>
    

This expects all machines in the `machine_list` to be reachable and to be 
set up with necessary code/libraries in `code_dir`. The user's code should also
be modified to add `CutPoint`s and use the `Varuna` training class.
The job is launched with all workers (`gpus_per_node x <num-servers>` in total) 
running the `user_training_script` with user args and arguments passed by varuna's launcher.
Any environment variables that the user wishes to pass to each worker may be specified 
in an `env_file` passed to the launcher.

These arguments passed by the launcher to the user training script for Varuna
must be parsed by user's training script and passed during `Varuna` initialisation:

* rank: process rank in overall distributed job
* local_rank: process rank in the local node 
* stage_to_rank_map: varuna config info about stage placement
* chunk_size: micro batch size for Varuna pipeline
* batch-size: per process batch size

The arguments for number of pipeline stages `nstages` and micro-batch size `chunk_size` can be
omitted if the user wishes Varuna to determine the most optimal configuration for these. 
This requires the user to run profiling before training and pass the location of stored 
profiles to the launcher. (see :doc:`profiler`)
