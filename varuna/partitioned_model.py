import torch
import torch.distributed as dist
from torch.nn import Module

import os, sys
import inspect
import time
import pickle

from .utils import save_rng_states, restore_rng_states, VARUNA_TEMP_FOLDER

from collections import OrderedDict 

class CutPoint(Module):

    def __init__(self):
        super(CutPoint, self).__init__()
        # start with 1 and end before last stage (total num_stages - 1 )
        self.cp_index = -1
        self.cp_func = None
        
        self.set_ret_val_func = None
        self.device = None
        self.send_fn = self.recv_fn = None
        self.stage = -1
        self.fp16 = False
        self.pruning = False
        self.barrier_event = None
        self.boundary_func = None
        self.forward_input_shapes = None
    
    def set_pruning(self, boolean):
        self.pruning = boolean

    def forward(self, *inputs, **kwargs):
        # not set by ModelParallel, pass through as is
        if self.barrier_event is not None:
            self.barrier_event.record()
        if self.boundary_func is not None:
            self.boundary_func()
        
        if self.cp_func is None:
            return inputs[0]

        if len(inputs) < 0 or (len(inputs) == 1 and inputs[0] is None):
            if self.pruning:
                inputs = (torch.tensor([-1.0], requires_grad = True))
            else:
                dtype = torch.float16 if self.fp16 else torch.float32
                inputs = torch.tensor([-1.0], requires_grad = True, dtype=dtype).to(self.device)
            # inputs.varuna_valid = False
            inputs = (inputs,)

        if isinstance(self.cp_func, torch.autograd.Function):
            out = self.cp_func.apply(*inputs, **kwargs)            
            if self.cp_index == (self.stage + 1):
                self.set_ret_val_func(out) 
            return out
        
        return self.cp_func(*inputs, **kwargs)

    def set_cp_func(self):
        
        is_in_next_stage = self.cp_index == self.stage
        is_in_prev_stage = self.cp_index == (self.stage + 1)

        class CutpointFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                # recieve activations
                if is_in_next_stage and self.recv_fn is not None:
                    i = self.recv_fn()
                # send activations
                elif is_in_prev_stage and self.send_fn is not None:
                    self.send_fn(i)
                return i

            @staticmethod
            def backward(ctx, grad_output):
                # receive gradients.
                if is_in_prev_stage and self.recv_fn is not None:
                    grad_output = self.recv_fn(grads = True)
                # send gradients
                elif is_in_next_stage and self.send_fn is not None:
                    self.send_fn(grad_output, grads = True)
                return grad_output
        
        c = CutpointFunction()
        self.cp_func = c


class PartitionedModel(Module):

    def __init__(self, module, rank, local_rank, device, stage_to_rank_map, fp16, shared_weights=None):
        super(PartitionedModel, self).__init__()
        self.module = module
        self.num_stages = len(stage_to_rank_map)
        self.stage_to_rank_map = stage_to_rank_map
        self.rank = rank
        self.local_rank = local_rank
        self.fp16 = fp16
        self.shared_weights = shared_weights

        
        self.grads_send_queue = self.acts_send_queue = None
        self.acts_queue = self.grads_queue = None
        
        torch.cuda.set_device(device)
        self.device = torch.device("cuda", device)

        self.ret_val = None
        self.pre_cp = None
        self.post_cp = None

        self.stage = -1
        for stage in self.stage_to_rank_map:
            if self.rank in self.stage_to_rank_map[stage]:
                self.stage = stage
                break
        else:
            raise ValueError("Rank " + self.rank + " not found in stage to rank map!")

        # self.logfile = open("wait_logs" + str(self.rank),"w")

    def initialize(self, dummy_inputs, from_cache=False):
        # print("Initializing partitioned model!")
        start = time.time()
        self.dry_run(dummy_inputs, from_cache)
        if self.shared_weights is not None:
            self.find_shared_weight_stages()
        print("dry run time", time.time() - start)
        self.prep_cutpoints()
        self.remove_unused_parameters(dummy_inputs)
        self.model_pruned = True


    def dry_run(self, dummy_inputs, from_cache):
        # """ executes the forward pass of the module on dummy inputs. Sets the order in which modules are used and the total number of cutpoints declared. """
        self.ordered_modules = OrderedDict()
        self.input_shapes = {}
        self.num_cutpoints = 0

        ord_mod_cache = os.path.join(VARUNA_TEMP_FOLDER, "_tmp_ord_mod" )
        inp_shape_cache = os.path.join(VARUNA_TEMP_FOLDER, "_tmp_inp_shapes")
        pstage_map_cache = os.path.join(VARUNA_TEMP_FOLDER, "_tmp_pstage_mapping")
        cache_present = os.path.exists(ord_mod_cache) and os.path.exists(inp_shape_cache) \
                        and os.path.exists(pstage_map_cache)

        if self.local_rank == 0 and not (from_cache and cache_present):
            # store input shapes for each module (or atleast each cp)
            print("Initializing partitioned model!")
  
            def get_hook(name):
                def add_module_hook(module, inputs, _output):
                    self.ordered_modules[name] = module
                    if isinstance(module, CutPoint):
                        self.input_shapes[name] = [list(inputs[0].size())]
                return add_module_hook

            param_access = dict()
            for p in self.module.parameters():
                param_access[p] = set()

            self.track_cp = 0
            
            def trace_param_access(frame, event, arg):
                if event != 'call':
                    return
                co = frame.f_code
                func_name = co.co_name
                if func_name in ['write','__hash__']:
                    return
                arg_info = inspect.getargvalues(frame)
                arg_values = [arg_info.locals[n] for n in arg_info.args]
                for arg in arg_values:
                    if isinstance(arg, torch.nn.Parameter):
                        if arg in param_access:
                            param_access[arg].add(self.track_cp)
            
            modules = self.module.named_modules()
            hooks = []

            def boundary_func():
                self.track_cp += 1
            for name, module in modules:
                if name == "":
                    continue
                hooks.append( module.register_forward_hook(get_hook(name)))
                if isinstance(module, CutPoint):
                    module.boundary_func = boundary_func
                    self.num_cutpoints += 1
            sys.settrace(trace_param_access)
            with torch.no_grad():
                self.module(**dummy_inputs)
            sys.settrace(None)
            self.track_cp = None
        
            for name in self.ordered_modules:
                m = self.ordered_modules[name]
                if isinstance(m, CutPoint):
                    m.boundary_func = None

            param_name_to_pstage = dict()
            for n,p in self.module.named_parameters():
                assert len(param_access[p]) < 2, f"Parameter {n} in multiple cuts: {param_access[p]}, mark as two shared parameters?"
                accesed_cps = list(param_access[p])
                if len(accesed_cps) > 0:
                    if n not in param_name_to_pstage:
                        param_name_to_pstage[n] = accesed_cps[0]
                    assert (param_name_to_pstage[n] == int(accesed_cps[0])), \
                            f"Parameter {n} accesed in cut {accesed_cps[0]} but was created in cut {param_name_to_pstage[n]}!"
            
            cp_index = 0
            modules = self.ordered_modules
            for name in modules:
                module = modules[name]
                if isinstance(module, CutPoint):
                    cp_index += 1
                    continue
                for n,p in module.named_parameters(recurse=False):
                    full_name = name + '.' + n
                    if full_name not in param_name_to_pstage:
                        param_name_to_pstage[full_name] = cp_index

            self.param_name_to_pstage = param_name_to_pstage

            for h in hooks:
                h.remove()

            if not os.path.exists(VARUNA_TEMP_FOLDER):
                os.makedir(VARUNA_TEMP_FOLDER)
            # actually just need the order as a list of names
            with open(ord_mod_cache,'wb') as f:
                pickle.dump(list(self.ordered_modules.keys()),f)
            with open(inp_shape_cache,'wb') as f:
                pickle.dump(self.input_shapes,f)
            with open(pstage_map_cache,'wb') as f:
                pickle.dump(self.param_name_to_pstage,f)
            dist.barrier()

        else:
            dist.barrier() 
            with open(ord_mod_cache,'rb') as f:
                ordered_modules = pickle.load(f)

            for n in ordered_modules:
                path = n.split(".")
                modules = self.module._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                self.ordered_modules[n] = modules[path[-1]]

            with open(inp_shape_cache,'rb') as f:
                self.input_shapes = pickle.load(f)
            with open(pstage_map_cache, 'rb') as f:
                self.param_name_to_pstage = pickle.load(f)
            self.num_cutpoints = len(self.input_shapes)
        
        # if self.local_rank == 0:
        #     for name in self.param_name_to_pstage:
        #         print(f"{name} {self.param_name_to_pstage[name]}\n", end = "")
    
    def find_shared_weight_stages(self):

        all_shared_weights = []
        for w_pair in self.shared_weights:
            all_shared_weights += [w for w in w_pair]
        curr_stage = 0
        weight_stages = dict()
        for m in self.ordered_modules:
            module = self.ordered_modules[m]
            if isinstance(module, CutPoint):
                curr_stage += 1
                continue
            for w in all_shared_weights:
                param_name = w.split(".")[-1]
                module_name = w[ : -len(param_name)-1]
                if m == module_name and hasattr(module, param_name):
                    weight_stages[w] = curr_stage
                    break
                elif m == module_name:
                    print("Here we have the peculiar case of the missing weight", m, param_name)
                    print(getattr(module,param_name))
        
        for w in all_shared_weights:
            if w not in weight_stages:
                param_name = w.split(".")[-1]
                if hasattr(self.module, param_name):
                    weight_stages[w] = curr_stage

        cuts_per_stage = (self.num_cutpoints + 1)/ self.num_stages
        shared_weight_stages = []
        for w_pair in self.shared_weights:
            for w in w_pair:
                assert w in weight_stages, "Shared parameter {} not found in model!".format(w)
                weight_stages[w] = int(weight_stages[w] // cuts_per_stage)
            shared_weight_stages.append((weight_stages[w_pair[0]], weight_stages[w_pair[1]]))

        self.shared_weight_stages = shared_weight_stages


# """ setting actual cutpoint functions for comunication. """
    def prep_cutpoints(self):

        def attach_meta(cutpoint, index):
            cutpoint.cp_index = index
            cutpoint.num_stages = self.num_stages
            cutpoint.set_ret_val_func = self.set_ret_val
            cutpoint.stage = self.stage
            cutpoint.device = self.device
            cutpoint.fp16 = self.fp16
            cutpoint.set_cp_func()

        self.cuts_per_stage = (self.num_cutpoints + 1) // self.num_stages

        modules = self.ordered_modules
        index = 1
        assigned_index = 1

        self.forward_input_shapes = []
        self.backward_grad_shapes = []

        for name in modules:
            module = modules[name]
            if name == "":
                continue
            if isinstance(module, CutPoint):
                if (index % self.cuts_per_stage == 0):
                    # pre cp
                    if assigned_index == self.stage:
                        self.forward_input_shapes = self.input_shapes[name]
                        self.pre_cp = module
                    # post cp
                    if assigned_index == self.stage + 1:
                        self.backward_grad_shapes = self.input_shapes[name]
                        self.post_cp = module
                    attach_meta(module, assigned_index)
                    assigned_index += 1  
                index += 1
            # found all relevant cutpoints, break
            if assigned_index == self.num_stages:
                break
        
# """ remove unused modules to save memory. """
    def remove_unused_parameters(self, dummy_inputs):

        pre_cp_index = self.stage
        post_cp_index = self.stage + 1

        is_used = {}
        used_modules = []
        add_flag = (self.stage == 0)
        
        modules = self.ordered_modules

        for name in modules:
            module = modules[name]
            if name == "":
                continue
            if isinstance(module, CutPoint):
                if (module.cp_index == pre_cp_index or module.cp_index == post_cp_index): 
                    add_flag = not add_flag
            else:
                if add_flag:
                    used_modules.append(name)
                is_used[name] = add_flag

        # any module that is used or has children that are used are needed
        for u in used_modules:
            path = u.split(".")
            key = path[0]
            for i in range(1,len(path)):
                is_used[key] = True
                key = key + "." + path[i]

        for m in is_used:
            if not is_used[m]:
                path = m.split(".")
                modules = self.module._modules
                for i in range(len(path) - 1):
                    modules = modules[path[i]]._modules
                modules[path[-1]] = None
                modules[path[-1]] = PassThroughModule()
                self.ordered_modules[m] = None

        self.check_unused_parameters(dummy_inputs)


    def parameter_names_to_cuts(self, dummy_inputs):
        
        return self.param_name_to_pstage

        stage_index = 0
        param_name_to_pstage = dict()
        
        print("NAME TO STAGE")
        
        for name in self.ordered_modules:
            module = self.ordered_modules[name]
            if module is None:
                continue
            if isinstance(module, CutPoint):
                stage_index += 1
            for n,p in module.named_parameters(recurse=False):
                param_name = name + '.' + n
                if param_name in param_name_to_pstage:
                    print(f"Duplicate param!! {param_name} {param_name_to_pstage[param_name]} {stage_index}")
                param_name_to_pstage[param_name] = stage_index
                # print(f"{self.stage} {param_name} , {stage_index}")

        # for n,p in self.module.named_parameters():
        #     assert n in param_name_to_pstage, f"{n} param not in stage {self.stage}"

        param_access = dict()
        for p in self.module.parameters():
            param_access[p] = set()

        self.track_cp = 0
        def boundary_func():
            self.track_cp += 1
        for n in self.ordered_modules:
            m = self.ordered_modules[n]
            if isinstance(m, CutPoint):
                m.boundary_func = boundary_func
        
        def trace_param_access(frame, event, arg):
            if event != 'call':
                return
            co = frame.f_code
            func_name = co.co_name
            if func_name in ['write','__hash__']:
                return
            arg_info = inspect.getargvalues(frame)
            arg_values = [arg_info.locals[n] for n in arg_info.args]
            for arg in arg_values:
                if isinstance(arg, torch.nn.Parameter):
                    if arg in param_access:
                        param_access[arg].add(self.track_cp)

        for k in dummy_inputs:
            if isinstance(dummy_inputs[k], torch.Tensor):
                dummy_inputs[k] = dummy_inputs[k].to(self.device)

        recv_fn = (lambda grads=False: torch.zeros(self.forward_input_shapes[0],  
                                        dtype=torch.float32, device=self.device) )    
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = recv_fn
                
        sys.settrace(trace_param_access)
        with torch.no_grad():
            try:
                self.module(**dummy_inputs)
            except Exception as e:
                if self.ret_val is None:
                    raise e
        self.ret_val = None
        sys.settrace(None)
        self.track_cp = None
        for n in self.ordered_modules:
            m = self.ordered_modules[n]
            if isinstance(m, CutPoint):
                m.boundary_func = None

        for n,p in self.module.named_parameters():
            assert len(param_access[p]) < 2, f"Parameter {n} in multiple cuts: {param_access[p]}, mark as shared parameters?"
            accesed_cps = list(param_access[p])
            if len(accesed_cps) > 0:
                if n not in param_name_to_pstage:
                    param_name_to_pstage[n] = accesed_cps[0]
                assert (param_name_to_pstage[n] == int(accesed_cps[0])), \
                         f"Parameter {n} accesed in cut {accesed_cps[0]} but was created in cut {param_name_to_pstage[n]}!"
                # print(f"{self.stage} {n} , {accesed_cps[0]}")      
        return param_name_to_pstage

    def check_unused_parameters(self, dummy_inputs):
        
        start_pstage = self.cuts_per_stage * self.stage
        end_pstage = self.cuts_per_stage * (self.stage+1)

        for n,p in self.module.named_parameters():
            if n not in self.param_name_to_pstage:
                # print(f"{n} not in pstage map")
                continue
            pstage = self.param_name_to_pstage[n]
            if pstage != -1 and (pstage < start_pstage or pstage >= end_pstage):
                # to_remove.append(n)
                path = n.split(".")
                parent = self.module
                for i in range(len(path) - 1):
                    parent = getattr(parent, path[i])
                setattr(parent,path[-1], None)

        self.model_pruned = True
           
    def set_ret_val(self, val):
        self.ret_val = val

    def set_queues(self, acts_send, grad_send, acts_recv, grad_recv, recompute):
        self.acts_send_queue = acts_send
        self.grads_send_queue = grad_send
        self.acts_queue = acts_recv
        self.grads_queue = grad_recv
        self.recompute_queue = recompute

    def set_send_fn(self, recompute = False):
        def send(tensor, grads = False):
            if grads:
                self.grads_send_queue.put(tensor.cpu())
            else:
                if not recompute:
                    self.acts_send_queue.put(tensor.cpu())

        if self.pre_cp is not None:
            self.pre_cp.send_fn = send
        if self.post_cp is not None:
            self.post_cp.send_fn = send

    def set_recv_fn(self, recompute=False):
        acts = None
        if recompute:
            ctx, acts = self.recompute_queue.get()
            restore_rng_states(ctx, self.device)
        else:
            acts = self.acts_queue.get() if self.stage > 0 else None
        if self.stage > 0:
            acts = acts.to(self.device) 

        def recv(grads = False):
            if grads:
                g = self.grads_queue.get().to(self.device)
                return g
            else:
                return acts
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = recv
        if self.post_cp is not None:
            self.post_cp.recv_fn = recv
        return acts

    def clear_recv_fn(self):
        if self.pre_cp is not None:
            self.pre_cp.recv_fn = None
        if self.post_cp is not None:
            self.post_cp.recv_fn = None

    def set_recording_events(self):
        self.recording_events = []
        if self.stage == 0:
            self.recording_events.append(torch.cuda.Event(enable_timing=True))
        in_stage = (self.stage == 0)
        for name in self.ordered_modules:
            module = self.ordered_modules[name]
            if isinstance(module, CutPoint):
                if module.cp_index == self.stage:
                    in_stage = True
                if in_stage:
                    event = torch.cuda.Event(enable_timing=True)
                    module.barrier_event = event
                    self.recording_events.append(event)
                if module.cp_index == self.stage + 1:
                    in_stage = False
        if self.stage == self.num_stages - 1:
            self.recording_events.append(torch.cuda.Event(enable_timing=True))

    def clear_recording_events(self):
        for name in self.ordered_modules:
            module = self.ordered_modules[name]
            if isinstance(module, CutPoint):
                module.barrier_event = None

    def elapsed_times(self):
        num_barriers = len(self.recording_events)
        times = []
        for i in range(num_barriers-1):
            times.append(
                self.recording_events[i].elapsed_time(self.recording_events[i+1])
                )
        return times

    def forward(self, inputs_as_dict, recompute=False, save_ctx=False, 
                recording=False, handle_comm=False):
        
        if save_ctx:
            # if these acts are going to be recomputed
            rng_states = save_rng_states(self.device)

        if recording:
            self.set_recording_events()
            if self.stage == 0:
                self.recording_events[0].record()

        if handle_comm:
            self.set_send_fn(recompute)
            recv_acts = self.set_recv_fn(recompute)
        try:
            calc_val = self.module(**inputs_as_dict)
            ret_val = self.ret_val if self.ret_val is not None else calc_val
        except Exception as e:
            if self.ret_val is None:
                raise e
            ret_val = self.ret_val
        self.ret_val = None

        if recording:
            if self.stage == self.num_stages - 1:
                self.recording_events[-1].record()
            self.clear_recording_events()
        
        if save_ctx:
            if self.stage > 0:
                recv_acts = recv_acts.cpu()
            ctx = (rng_states, recv_acts)
            self.recompute_queue.put(ctx)

        if handle_comm:
            self.clear_recv_fn()

        return ret_val 


class PassThroughModule(Module):

    def __init__(self):
        super(PassThroughModule, self).__init__()

    def forward(self,*args,**kwargs):
        return None


