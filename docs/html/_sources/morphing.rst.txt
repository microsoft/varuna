Morphing
========

Varuna enables distributed training on a changing set of resources, as the list
of machines available may grow or shrink. This is done by "morphing" - reconfiguring the 
training job to process the total effective batch size over the new resources. Varuna performs
morphing by checkpointing and restarting efficiently, which requires that the training job has access to
a long-living 'manager' machine and a global storage for all workers.

The manager launches the `run_varuna` command, detects changes in the available resource set, slow GPUs
or transient errors in the job, and cooridinates checkpoint/restarts. If desirable, the manager 
can be notified of an upcoming preemption (loss of a machine) through the function `notify_manager_preempt`.
For example in Azure, a 'preempt' signal is issued with preemption time.

To enable morphing, the user must make some modifications to their script:

* An additional `resume_step` argument is passed to each worker for restarts. (So that there 
    are no race conditions while checking this step from the global storage)
* A simple signal handler for `SIGUSR1` in the workers to call varuna's `on_demand_checkpoint` (:doc:`varuna`)
    and exit. The checkpointing may fail if workers are lost during the call.
* (recommended) With morphing, `Varuna` checkpointing should be enabled with background copying and sharding flags for 
    faster checkpointing. The checkpoint frequency should be high to avoid loss of compute on checkpoint/restarts 
    (in case on demand checkpoints fail).

These changes are illustrated in the megatron example.

The key idea behind morphing is to re-distribute the total `batch_size` specified by the user accross
pipeline parallel stages and data parallel replicas. To do this efficiently, it is recommended to use
auto-configuration of the dimensions of pipeline and data parallelism as well as the micro-batch size.
`AutoConfig` by varuna is enabled if these arguments (`nstages` and `chunk_size`) are not specified 
while launching `run_varuna`. This estimates the best varuna configuration at each point and requires 
the user to run profiling before training and specify the location of stored profiles to the 
launcher. (see :doc:`profiler`)

==================
Slow GPU detection
==================

With low-priority VMs, a user might see faulty "straggler" GPUs that have significantly longer compute
times than the others. These are detected by varuna when morphing is enabled by the manager.
The IPs with the slow GPUs are written to a file "slow_machines.out". The user may listen on this file 
to remove machines with faulty GPUs. 
