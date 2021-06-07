# to be run in manager
import socket
import threading
from threading import Thread
import socketserver
import time
from datetime import datetime, timedelta
import os
import subprocess
import sys
from varuna import AutoConfig

total_num_iterations = 18750
batch_size = 8192
ckpt_dir = "/home/varuna/gpt2-blob/mega_2_5b_4.8e-3_clipgrads"
GPUS_PER_VM = 1

checkpointed = -1
is_preempting = False
is_restarting = False
is_morphing = False
last_ckpt_signal = None
curr_world_size = 0
last_preempt_handled = None
last_restart_time = datetime.now() - timedelta(hours=2)
consecutive_restarts = 0
if len(sys.argv) > 1:
    curr_world_size = int(sys.argv[1])
if len(sys.argv) > 2:
    checkpointed = int(sys.argv[2])
bad_ips = []

class Handler(socketserver.BaseRequestHandler):

    triggermorph = threading.Lock()
    scripts_folder = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

    @staticmethod
    def update_available():
        print("updating available", flush=True)
        os.system("bash {} > {}".format(\
                os.path.join(Handler.scripts_folder,"get_available_machines.sh"), \
                os.path.join(Handler.scripts_folder, "available_machines.out")))

    @staticmethod
    def send_signal():
        print("sending signal", flush = True)
        sh = os.path.join(Handler.scripts_folder, "send_signal.sh")
        p = None
        try:
            p = subprocess.call(['bash', sh], timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print("signal timed/errored out: ",e)
            if p is not None:
                p.kill()

    @staticmethod
    def kill_all():
        print("killing all", flush=True)
        sh = os.path.join(Handler.scripts_folder, "kill_all.sh")
        p = None
        try:
            p = subprocess.call(['bash', sh], timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print("kill errored/timed out: ", e)
            if p is not None:
                p.kill()

    @staticmethod
    def start_remote(resume=-1):
        print("restarting", resume, flush=True)
        os.system("bash {} {}".format( \
            os.path.join(Handler.scripts_folder, "start_remote.sh"), resume))

    @staticmethod
    def get_available():
        filename = os.path.join(Handler.scripts_folder, "available_machines.out")
        f = open(filename,"r")
        machines = f.read().split("\n")
        if machines[-1] == "":
            machines = machines[:-1]
        return machines
    
    def handle(self):
        global checkpointed, is_preempting, is_restarting, is_morphing, \
            last_ckpt_signal, curr_world_size, last_preempt_handled, \
            total_num_iterations, last_restart_time, consecutive_restarts, bad_ips, batch_size
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        # print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)
        
        if 'starting' in data:
            Handler.triggermorph.acquire()
            try:
                curr_world_size = int(data.split(" ")[-1])
                print("\nStarted job with world size", curr_world_size, flush=True)
            except Exception as e:
                print("Caught Exception while starting", e)
            Handler.triggermorph.release()
        
        if 'already running' in data:
            Handler.triggermorph.acquire()
            try:
                curr_world_size = int(data.split(" ")[-1])
                print("\nA job is already running, setting world size", curr_world_size, flush=True)
            except Exception as e:
                print("Caught Exception while starting", e)
            Handler.triggermorph.release()

        elif 'preempt' in data:
            Handler.triggermorph.acquire()
            try:
                print(f"preempt signal from {self.client_address[0]}:",end="")
                if not is_morphing and not is_preempting and not is_restarting:
                    fields = data.split(" ")
                    if len(fields) > 1:
                        notbefore = fields[-1]
                        notbefore = datetime.strptime(notbefore,"%a,_%d_%b_%Y_%H:%M:%S_%Z")
                    else:
                        notbefore = datetime.now()
                    if last_preempt_handled is None or last_preempt_handled < notbefore:
                        last_preempt_handled = notbefore
                        is_preempting = True  
                        sleep_time = (notbefore - datetime.now()).seconds - 60
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        print('Trigger preempt!', flush=True)
                        Handler.send_signal()
                    else:
                        print("this preempt was already handled or has passed")
                else:
                    print('preempt already triggered!',flush=True)
            except Exception as e:
                print("Caught exception while preempting:", e,flush=True)
                is_preempting = False
            Handler.triggermorph.release()
        
        elif 'checkpoint done' in data:
            Handler.triggermorph.acquire()
            try:
                last_iter = int(str(data).split(" ")[-1])
                print(f"ckpt done {last_iter}:",end="")
                if last_iter >= total_num_iterations:
                    print("\nTraining complete! Exiting")
                    exit()
                if last_iter > checkpointed:
                    checkpointed = last_iter
                if is_preempting:
                    print('\nPreempt successful {}'.format(last_iter), flush=True)
                    time.sleep(120)     # wait for scheduled event to occur 
                    Handler.update_available()
                    Handler.kill_all()
                    curr_world_size = 0
                    Handler.start_remote(checkpointed)
                    is_preempting = False
                elif is_morphing:
                    print("\nMorph successful {}".format(last_iter), flush=True)
                    Handler.kill_all()
                    curr_world_size = 0
                    Handler.start_remote(checkpointed)
                    is_morphing = False
                    is_restarting = False
                elif not is_restarting and \
                    last_ckpt_signal is None or \
                    (recv_time - last_ckpt_signal).total_seconds() > 10:
                    if consecutive_restarts >= 3:
                        print("\nToo many consecutive restarts, waiting...",flush=True)
                        consecutive_restarts = 0
                        # Handler.send_signal()
                        # Handler.kill_all()
                        is_restarting = False
                        curr_world_size = 0
                    else:
                        print("\nHandling restart", last_ckpt_signal, flush=True)
                        last_iter = int(str(data).split(" ")[-1])
                        time.sleep(120)    # wait for transient errors to pass
                        Handler.update_available()
                        Handler.kill_all()
                        curr_world_size = 0
                        Handler.start_remote(checkpointed)
                        is_restarting = False
                        restart_time = datetime.now()
                        if (restart_time - last_restart_time).total_seconds() < 300:
                            consecutive_restarts += 1
                        else:
                            consecutive_restarts = 0
                        last_restart_time = restart_time
            except Exception as e:
                is_restarting = False
                is_morphing = False
                is_preempting = False
                print("Caught exception after ckpt", e, flush=True)
            last_ckpt_signal = recv_time
            Handler.triggermorph.release()
        
        elif 'morph' in data:
            Handler.triggermorph.acquire()
            try:
                new_availability = int(data.split(" ")[-1])
                print("morph signal!", new_availability)
                my_wd = os.getcwd()
                os.chdir(Handler.scripts_folder)
                running_machines = Handler.get_available()[:curr_world_size]
                Handler.update_available()
                new_machines = Handler.get_available()
                new_availability = len(new_machines) * GPUS_PER_VM
                old_preempted = False
                for m in running_machines:
                    if m not in new_machines:
                        old_preempted = True
                        break
                print("old was preempted? ", old_preempted)
                if new_availability > 0:
                    auto = AutoConfig(new_availability, GPUS_PER_VM, batch_size, verbose=False)
                    num_partitions, _, _ = auto.get_min()
                    dp_size = new_availability // num_partitions
                    print("best config is", num_partitions, "x", dp_size)
                    new_world_size = dp_size * num_partitions
                else:
                    new_world_size = 0
                os.chdir(my_wd)
                if (not old_preempted) and new_world_size == curr_world_size:
                    print("World size will still be", curr_world_size)
                    print("not morphing", flush=True)
                else: 
                    if not is_preempting and not is_restarting and not is_morphing:
                        print("Morphing!",flush=True)
                        is_restarting = True
                        is_morphing = True
                        response = Handler.send_signal()
                        if curr_world_size == 0:
                            print("Nothing running currently, will start")
                            Handler.kill_all()
                            Handler.start_remote(checkpointed)
                            is_morphing = False
                            is_restarting = False
                    else:
                        print("morph change was already detected", is_morphing, is_preempting, is_restarting)
            except Exception as e:
                print("Caught exception while morphing:", e, flush=True)
                is_morphing = False
                is_restarting = False
            Handler.triggermorph.release()
        elif 'no nvidia' in data:
            print(f"{self,client_address[0]} does not have nvidia")
            print(data)
            ip = self.client_address[0]
            if ip not in bad_ips:
                print("rebooting")
                os.system(f"ssh {ip} 'sudo reboot'")
                bad_ips.append(ip)
            else:
                print("shutting down")
                os.system(f"ssh {ip} 'sudo shutdown'")
                bad_ips.remove(ip)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "10.0.0.4", 4200

    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    with server:
        server.serve_forever()