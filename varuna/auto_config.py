
import math
import os
import torch
import pickle

class AutoConfig:

    def __init__(self, num_gpus, gpus_per_vm, batch_size,
                profile_folder, gpu_memory_capacity=None, verbose=True, 
                autofill_missing_compute=False):

        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.gpus_per_vm = gpus_per_vm
        if gpu_memory_capacity is None:
            gpu_memory_capacity = torch.cuda.get_device_properties(0).total_memory
        self.gpu_memory_capacity = gpu_memory_capacity
        
        self.read_model_structure()
        self.read_profile(profile_folder, autofill_missing_compute)

        num_stages_candidates = [ i for i in range(1, self.num_pstages) if self.num_pstages % i == 0]
        self.batch_times = dict()
        self.micro_batch = dict()

        for pp_size in num_stages_candidates:
            if num_gpus < pp_size:
                print(f"can't have {pp_size} stages!")
                continue
            if verbose:
                print("Stages", pp_size)
            # get highest micro batch for each num_stage_cand
            mbs = self.get_microbatch_size(pp_size)
            print(f"Predicted microbatch size for {pp_size}: {mbs}")
            self.micro_batch[pp_size] = mbs
            dp_size = num_gpus // pp_size
            cuts_per_stage = self.num_pstages // pp_size
            num_microbatches = math.ceil((batch_size // dp_size ) / mbs)

            self.calc_and_write_compute_times(pp_size, mbs)
            # TODO: comm profile for last cp in stage for each stage
            # TODO: comm profile missing send/long-send
            if pp_size > 1:
                comm_size = mbs
                for d in self.input_shapes[cuts_per_stage]:
                    comm_size *= d
            else:
                comm_size = 0
            print("comm size", comm_size)
            send_time = self.comm_profile[comm_size]["send"]
            long_send_time = self.comm_profile[comm_size]["long_send"]
            if send_time == -1:
                print(f"WARNING: no send time found, {pp_size} partitions")
                send_time = 0
            if long_send_time == -1:
                print(f"WARNING: no long send time found, {pp_size} partitions, size {comm_size}, {long_send_time}")
                long_send_time = 0
            alr = self.get_alr_time(dp_size, pp_size)
            if alr == -1:
                print(f"WARNING: no allreduce time found for {pp_size} x {dp_size}")
                alr = 0
            
            batch_time = self.get_simulated_time(pp_size, num_microbatches, send_time, \
                                long_send_time, alr, verbose)
            self.batch_times[pp_size] = batch_time

        print(self.batch_times)
        print(self.micro_batch)
 
    def calc_and_write_compute_times(self, pp_size, mbs):
        pstages_per_stage = self.num_pstages // pp_size

        fwd_times = []
        bwd_times = []
        out = open("compute.txt", "w")
        for stage in range(pp_size):
            fwd_time = 0.0; bwd_time = 0.0
            pstages = range(pstages_per_stage*stage, pstages_per_stage*(stage+1))
            for pstage in pstages:
                fwd_time += self.compute_profile[pstage][mbs]["fwd"]
                bwd_time += self.compute_profile[pstage][mbs]["bwd"]

            # TODO: COPY OR NO COPY !!!
            # # acts copy
            # if stage < (pp_size-1):
            #     copy = self.compute_profile[pstages_per_stage*(stage+1) - 1][mbs]["copy"]
            #     fwd_time += copy; bwd_time += copy

            # # grads copy
            # if stage > 0:
            #     copy = self.compute_profile[pstages_per_stage * stage][mbs]["copy"]
            #     fwd_time += copy; bwd_time += copy
            
            fwd_time = int(fwd_time)
            bwd_time = int(bwd_time)
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)

            out.write(f"{fwd_time} {bwd_time}\n")
            # print(f"{fwd_time} {bwd_time}")

        out.close()

    def get_simulated_time(self, pp_size, num_microbatches, send_time, \
                    long_send_time, alr, verbose=False):
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'tools')
        sim_binary = os.path.join(tools_dir, "simulator", "simulate-varuna")
        # TODO: use C++ extension for python?
        command = f"GPUS_PER_VM={self.gpus_per_vm} {sim_binary} " + \
                    f"{pp_size} {num_microbatches} {send_time} {alr} {long_send_time}"
        if verbose:
            print(command)
        simulate = os.popen( command).read()
        if verbose:
            print(simulate)
        batch_time = simulate.split("\n")[0]
        batch_time = int(batch_time.split(" ")[-1])
        batch_time = batch_time / 1000000
        return batch_time


    def get_min(self):
        min_time = 1000000000000
        best_pp = -1; best_mbs = -1
        for pp_size in self.batch_times:
            if min_time > self.batch_times[pp_size]:
                best_pp = pp_size
                best_mbs = self.micro_batch[pp_size]
                min_time = self.batch_times[pp_size]
        return best_pp, best_mbs, min_time

    def read_profile(self, profile_folfder, autofill_missing_compute=False):
        
        self.compute_profile = []
        self.comm_profile = []
        for i in range(self.num_pstages):
            profile_path = os.path.join(profile_folfder, f"compute-profile-{i}")
            if os.path.exists(profile_path):
                with open(profile_path, "rb") as f:
                    compute_profile = pickle.load(f)
            else:
                assert autofill_missing_compute and i>0, \
                "Missing compute profiles! Profile should have compute for at least one cutpoint." + \
                "Enable flag autofill_missing_compute if some others are missing."
                compute_profile = self.compute_profile[0]         
            self.compute_profile.append(compute_profile)

        with open(os.path.join(profile_folfder, f"comm-profile"), "rb") as f:
            self.comm_profile = pickle.load(f)

        with open(os.path.join(profile_folfder, "allred-profile"), "rb") as f:
            self.all_reduce_profile = pickle.load(f)

    def read_model_structure(self):
        with open("_tmp_inp_shapes",'rb') as f:
            input_shapes = pickle.load(f)
        input_shapes_keys = list(input_shapes.keys())
        self.input_shapes = [input_shapes[k][0] for k in input_shapes_keys]
        with open("_tmp_shape_changes",'rb') as f:
            shape_indices_to_change = pickle.load(f)
        self.shape_indices_to_change = [shape_indices_to_change[k][0] for k in input_shapes_keys]
        self.num_pstages = len(self.input_shapes) + 1
        if verbose:
            print(self.num_pstages,"cutpoints")

    def get_alr_time(self, dp_size, pp_size):
        if dp_size < 2:
            alr = 0
        elif len(self.all_reduce_profile[pp_size]) < 2:
            return -1
        elif dp_size <= len(self.all_reduce_profile[pp_size]):
            alr = self.all_reduce_profile[pp_size][dp_size - 1]
        else:
            # TODO: is this acceptable? 16 vs 31 ring size?
            closest_power = 2**math.floor(math.log(dp_size,2))
            alr = self.all_reduce_profile[pp_size][closest_power-1] \
                if closest_power <= len(self.all_reduce_profile[pp_size]) \
                else self.all_reduce_profile[pp_size][1] * math.log(dp_size, 2)
        return alr

    def get_microbatch_size(self, pp_size):
        pstages_per_stage = self.num_pstages // pp_size

        def get_max_mem(mbs):
            max_memory_used = 0
            for stage in range(pp_size):
                pstages = range(pstages_per_stage*stage, pstages_per_stage*(stage+1))
                mem_usage = 0
                for pstage in pstages:
                    mem_usage += ( self.compute_profile[pstage][mbs]["max_memory"] - \
                                self.compute_profile[pstage][mbs]["acts_size"] )
                last_cp = ( pstages_per_stage * (stage + 1) ) - 1
                mem_usage += self.compute_profile[last_cp][mbs]["acts_size"]
                max_memory_used = max(mem_usage,max_memory_used)
            return max_memory_used

        max_micro_bs = max([len(profile) for profile in self.compute_profile])
        start = 1; end = max_micro_bs
        limit = self.gpu_memory_capacity

        while start < end:
            mid = int(math.ceil((start+end) / 2))
            mem_usage = get_max_mem(mid)
            if mem_usage > limit:
                end = mid-1
            elif mem_usage < limit:
                start = mid
            else:
                start = mid
                end = mid
        
        # assert start == end,f"No microbatch fits for {pp_size} partitions!"
        if start == end:
            return start
        return -1