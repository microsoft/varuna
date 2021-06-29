# to be run in manager
import socket
import threading
from threading import Thread
import socketserver
import time
from datetime import datetime
import os
import subprocess
import sys

checkpointed = -1
is_preempting = False
is_restarting = False
is_morphing = False
last_ckpt_signal = None
curr_world_size = 0
last_iter = -1
progress_iter = 0
last_preempt_handled = None

# if len(sys.argv) > 1:
#     curr_world_size = int(sys.argv[1])
# if len(sys.argv) > 2:
#     checkpointed = int(sys.argv[2])

available_machines_list = sys.argv[1]
running_machines_list = sys.argv[2]     # TODO coord this with run_varuna
PORT = int(sys.argv[3])
my_ip = socket.gethostbyname(socket.gethostname())
HOST = my_ip

class Handler(socketserver.BaseRequestHandler):

    triggermorph = threading.Lock()
    scripts_folder = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def update_available():
        global available_machines_list, running_machines_list
        print("updating available", flush=True)
        with open(available_machines_list, "r") as inf:
            with open(running_machines_list, "r") as outf:
                outf.write(f.read())

    @staticmethod
    def send_signal():
        global running_machines_list
        print("sending signal", flush = True)
        sh = os.path.join(Handler.scripts_folder, "send_signal.sh")
        p = None
        try:
            p = subprocess.call(['bash', sh, running_machines_list], timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print("signal timed/errored out: ",e)
            if p is not None:
                p.kill()

    @staticmethod
    def kill_all():
        global running_machines_list
        print("killing all", flush=True)
        sh = os.path.join(Handler.scripts_folder, "kill_all.sh")
        p = None
        try:
            p = subprocess.call(['bash', sh, running_machines_list], timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print("kill errored/timed out: ", e)
            if p is not None:
                p.kill()

    @staticmethod
    def start_remote(resume=-1):
        global available_machines_list, my_ip
        print("restarting", resume, flush=True)
        cmd = "python -m varuna.run_varuna --resume " + \
             f"--machine_list {available_machines_list} --manager_ip {my_ip}"
        os.system(cmd)
    
    def handle(self):
        global checkpointed, is_preempting, is_restarting, is_morphing, last_ckpt_signal, curr_world_size, last_iter, last_preempt_handled
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)
        
        if 'is_running?' in data:
            response = bytes("yes", 'ascii')
            self.request.sendall(response)

        if 'starting' in data:
            Handler.triggermorph.acquire()
            print("Lock acquired by start:", is_restarting, is_morphing, is_preempting, flush=True)
            try:
                curr_world_size = int(data.split(" ")[-1])
                print("Started job with world size", curr_world_size)
            except Exception as e:
                print("Caught Exception while starting", e)
            Handler.triggermorph.release()
            print("Lock released by start:", is_restarting, is_morphing, is_preempting)
        
        elif 'preempt' in data:
            Handler.triggermorph.acquire()
            print("Lock acquired by preempt:", is_restarting, is_morphing, is_preempting, flush=True)
            try:
                if not is_morphing and not is_preempting and not is_restarting:
                    fields = data.split(" ")
                    notbefore = fields[-1]
                    notbefore = datetime.strptime(notbefore,"%a,_%d_%b_%Y_%H:%M:%S_%Z")
                    if last_preempt_handled is None or last_preempt_handled < notbefore:
                        last_preempt_handled = notbefore
                        is_preempting = True  
                        sleep_time = (notbefore - datetime.now()).seconds - 30
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        print('Trigger preempt!', flush=True)
                        Handler.send_signal()
                    else:
                        print("this preempt was already handled or has passed")
                else:
                    print('preempt already triggered!',flush=True)
            except Exception as e:
                print("Caught exception while preempting:", e)
                is_preempting = False
            Handler.triggermorph.release()
            print("Lock released by preempt:", is_restarting, is_morphing, is_preempting)
        
        elif 'checkpoint done' in data:
            Handler.triggermorph.acquire()
            print("Lock acquired by ckpt:", recv_time, is_restarting, is_morphing, is_preempting, flush=True)
            try:
                last_iter = int(str(data).split(" ")[-1])
                if last_iter > checkpointed:
                    checkpointed = last_iter
                if is_preempting:
                    print('Preempt successful {}'.format(last_iter), flush=True)
                    time.sleep(120)     # wait for scheduled event to occur 
                    Handler.kill_all()
                    curr_world_size = 0
                    # Handler.update_available()
                    Handler.start_remote(checkpointed)
                    is_preempting = False
                elif is_morphing:
                    print("Morph successful {}".format(last_iter), flush=True)
                    Handler.kill_all()
                    curr_world_size = 0
                    Handler.start_remote(checkpointed)
                    is_morphing = False
                    is_restarting = False
                elif not is_restarting:
                    if last_ckpt_signal is None or \
                    (recv_time - last_ckpt_signal).total_seconds() > 100:
                        print("Handling restart", last_ckpt_signal)
                        last_iter = int(str(data).split(" ")[-1])
                        time.sleep(60)    # wait for transient errors to pass
                        Handler.kill_all()
                        curr_world_size = 0
                        # Handler.update_available()
                        Handler.start_remote(checkpointed)
                        is_restarting = False
            except Exception as e:
                is_restarting = False
                is_morphing = False
                is_preempting = False
                print("Caught exception after ckpt", e)
            last_ckpt_signal = recv_time
            Handler.triggermorph.release()
            print("Lock released by ckpt done:", is_restarting, is_morphing, is_preempting)
        
        elif 'morph' in data:
            Handler.triggermorph.acquire()
            print("Lock acquired by morph:", is_restarting, is_morphing, is_preempting, flush=True)
            try:
                if not is_preempting and not is_restarting and not is_morphing:
                    print("Morphing!",flush=True)
                    is_restarting = True
                    is_morphing = True
                    response = Handler.send_signal()
                    # Handler.update_available()
                    if curr_world_size == 0:
                        print("Nothing running currently, will start")
                        Handler.kill_all()
                        Handler.start_remote(checkpointed)
                        is_morphing = False
                        is_restarting = False
                else:
                    print("morph change was already detected", is_morphing, is_preempting, is_restarting)
            except Exception as e:
                print("Caught exception while morphing:", e)
                is_morphing = False
                is_restarting = False
            Handler.triggermorph.release()
            print("Lock released by morph:", is_restarting, is_morphing, is_preempting)

        elif 'progress' in data:
            global progress_iter
            Handler.triggermorph.acquire()
            progress_iter = int(data.split(" ")[-1].strip())
            Handler.triggermorph.release()
        print("handle done for", data, recv_time,  flush=True)
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    with server:
        server.serve_forever()
