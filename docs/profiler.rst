Profiling for Varuna
====================

The Varuna :class:`Profiler` provides an easy interface for users to profile 
the compute and communication operations of a model. This processes the model cutpoints in parallel
and captures the time and memory consumption for each cutpoint. This profile can then be used to calculate
various parameters for Varuna - ideal pipeline and data-parallel dimensions for a given number of GPUs and
suitable microbatch sizes for different configs.

.. py:currentmodule:: varuna

.. autoclass:: Profiler
   
   .. automethod:: set_optimizer
   .. automethod:: profile_all