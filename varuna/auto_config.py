
import math
import os

micro_batch = dict({27:	12,
                    18:	10,
                    9:	2,
                    6:	2})

alr_time = dict({54: [0.483465296, 0.74187516, 0.664803532, 0.842620478, 1.232588885, 0.815255536, 1.267308886, 1.237618767, 1.022943954],
    27: [0.756439877, 1.171520746, 1.011008424, 1.172397685, 1.815907892, 1.24537253, 1.928181718, 2.051309269, 1.773356295],
    18: [1.111971784, 1.754020047, 1.511168528, 1.715411258, 2.740385453, 1.953130831, 2.946040092, 3.092466296, 2.316881182],
    9: [2.180923504, 3.772105455, 3.186639357, 3.62031307, 5.335995863, 3.758462596, 6.098021951, 5.818171277, 4.868596537],
    6: [3.190522587, 4.670527804, 4.382830995, 5.550135493, 8.085498104, 5.589711748, 8.964560346, 9.211063372, 7.290586735],
    3: [5.194916379, 7.58669198, 7.838974547, 9.698126502, 15.76477601, 10.41937665, 15.57246651, 17.65429022, 11.63799074]})


send = dict({6:	0.03249843472028634,
            9:	0.0330769070571031,	
            18:	0.16968193460018077,	
            27:	0.21019042793073153})

long_send = dict({6:	0.152079960,
                9:	0.1520799069,
                18:	0.620104318,
                27:	0.721766365})

class AutoConfig:

    def __init__(self, num_gpus, gpus_per_vm, batch_size, verbose=True):

        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.gpus_per_vm = gpus_per_vm
        
        comp0 = self.read_profile("profiles/gpt2-2_5bnew-stage0")
        comp1 = self.read_profile("profiles/gpt2-2_5bnew-stage1")
        complast = self.read_profile("profiles/gpt2-2_5bnew-stage53")

        compute_profile = [comp0]
        for i in range(52):
            compute_profile.append(comp1)
        compute_profile.append(complast)

        num_pstages = len(compute_profile)
        num_stages_candidates = [ i for i in range(1, num_pstages) if num_pstages % i == 0]
        num_stages_candidates = [6,9]
        self.batch_times = dict()

        for pp_size in num_stages_candidates:
            if num_gpus < pp_size:
                print(f"can't have {pp_size} stages!")
                continue
            if verbose:
                print("Stages", pp_size)
            # get highest micro batch for each num_stage_cand
            mbs = micro_batch[pp_size]
            dp_size = num_gpus // pp_size
            num_microbatches = math.ceil((batch_size // dp_size ) / mbs)
            pstages_per_stage = num_pstages // pp_size
            fwd_times = []
            bwd_times = []
            out = open("compute.txt", "w")
            for stage in range(pp_size):
                fwd_time = 0.0; bwd_time = 0.0
                for pstage in range(pstages_per_stage*stage, pstages_per_stage*(stage+1)):
                    fwd_time += compute_profile[pstage][mbs]["fwd"]
                    bwd_time += compute_profile[pstage][mbs]["bwd"]
                # acts copy
                if stage < (pp_size-1):
                    copy = compute_profile[pstages_per_stage*(stage+1)-1][mbs]["copy"]
                    fwd_time += copy; bwd_time += copy
                # grads copy
                if stage > 0:
                    copy = compute_profile[pstages_per_stage*stage][mbs]["copy"]
                    fwd_time += copy; bwd_time += copy
                fwd_time = int(fwd_time)
                bwd_time = int(bwd_time)
                fwd_times.append(fwd_time)
                bwd_times.append(bwd_time)
                out.write(f"{fwd_time} {bwd_time}\n")
            out.close()
            send_time = send[pp_size] * 1000000
            long_send_time = long_send[pp_size] * 1000000
            if dp_size == 1:
                alr = 0
            elif dp_size - 2 < len(alr_time[pp_size]):
                alr = alr_time[pp_size][dp_size - 2]
            else:
                closest_power = 2**math.floor(math.log(dp_size,2))
                alr = alr_time[pp_size][closest_power-2] \
                    if closest_power - 2 < len(alr_time[pp_size]) \
                    else alr_time[pp_size][0] * math.log(dp_size,2)
            alr *= 1000000
            tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'tools')
            sim_binary = os.path.join(tools_dir, "simulator", "simulate-varuna")
            command = f'GPIPE=0 SCATTERED=0 GPUS_PER_VM={self.gpus_per_vm} {sim_binary} \
                        {pp_size} {num_microbatches} {send_time} {alr} {long_send_time}'
            if verbose:
                print(command)
            simulate = os.popen( command).read()
            if verbose:
                print(simulate)
            batch_time = simulate.split("\n")[0]
            batch_time = int(batch_time.split(" ")[-1])
            batch_time = batch_time / 1000000
            self.batch_times[pp_size] = batch_time

        print(self.batch_times)
    
    def get_min(self):
        min_time = 1000000000000
        best_pp = -1; best_mbs = -1
        for pp_size in self.batch_times:
            if min_time > self.batch_times[pp_size]:
                best_pp = pp_size
                best_mbs = micro_batch[pp_size]
                min_time = self.batch_times[pp_size]
        return best_pp, best_mbs, min_time

    def read_profile(self, filename):
        compute = dict()
        with open(filename, "r") as f:
            for line in f:
                mbs, fwd, bwd, copy = line.split(" ")
                mbs = int(mbs)
                fwd = float(fwd) * 1000000
                bwd = float(bwd) * 1000000
                copy = float(copy) * 1000000
                compute[mbs] = {"fwd":fwd, "bwd":bwd, "copy": copy}
        return compute