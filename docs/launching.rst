Launching Varuna
================

Training with varuna can be run with the run_varuna module as follows:

.. code-block:: bash

    python -m varuna.run_varuna --machine_list <file_with_ips> --gpus_per_node <num_gpus_per_node> 
    --batch-size <total_effective_batch_size> --nstages <number_of_pipeline_stages> 
    --chunk_size <micro_batch_size_for_pipeline> 
    --code_dir <working_dir_for_training> user_training_script.py <...user args...>
    

This expects all machines in the machine_list to be reachable from the launching machine, to be 
set up with necessary code/libraries in code_dir and have gpus_per_node GPUs working. 
The job is launched with all workers running the user_training_script and args.

This launcher passes a few arguments to the user training script for Varuna. These should be passed during `Varuna` initialisation in the python script:
* rank: process rank in overall distributed job
* local_rank: process rank in the local node 
* stage_to_rank_map: varuna config info about stage placement
* chunk_size: micro batch size for Varuna pipeline

The arguments for number of pipeline stages (nstages) and micro-batch size (chunk_size) can be
omitted if the user wishes Varuna to determine the most optimal configuration for these. 
This requires the user to run profiling before training and pass the location of stored profiles to the launcher.
  
