# to be run in manager
import socket
import threading
import socketserver
import time
from datetime import datetime
import os

checkpointed = 0
is_preempting = False
is_restarting = False
is_morphing = False
last_ckpt_signal = None
curr_world_size = 0
last_iter = -1

class Handler(socketserver.BaseRequestHandler):

    triggermorph = threading.Lock()
    trackcheckpoints = threading.Lock()
    scripts_folder = "/home/varuna/t-saathl/mega1_5b/Megatron-LM/"

    @staticmethod
    def update_available():
        print("updating available")
        os.system("bash {} > {}".format(\
                os.path.join(Handler.scripts_folder,"get_available_machines.sh"), \
                os.path.join(Handler.scripts_folder, "available_machines.out")))

    @staticmethod
    def send_signal():
        print("sending signal")
        return os.system("bash {}".format(os.path.join(Handler.scripts_folder, "send_signal.sh"))) 

    @staticmethod
    def kill_all():
        print("killing all")
        os.system("bash {}".format(os.path.join(Handler.scripts_folder, "kill_all.sh")))

    @staticmethod
    def start_remote(resume=-1):
        print("restarting", resume)
        os.system("bash {} {}".format( \
            os.path.join(Handler.scripts_folder, "start_remote.sh"), resume))

    @staticmethod
    def notify():
        os.system('( echo "Subject: Job morphing";\
                     echo "machines were preempted") |\
                     ssmtp -F "Varuna" "nitika.saran@gmail.com, \
                     t-saathl@microsoft.com"')


    def handle(self):
        global checkpointed, is_preempting, is_restarting, is_morphing, last_ckpt_signal, curr_world_size, last_iter
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)
        
        if 'starting' in data:
            Handler.triggermorph.acquire()
            curr_world_size = int(data.split(" ")[-1])
            print("Started job with world size", curr_world_size)
            Handler.triggermorph.release()
        
        elif 'preempt' in data:
            Handler.triggermorph.acquire()
            if not is_morphing and not is_preempting and not is_restarting:
                # set False to ignore signals from other VMs, set True after checkpointing succeeds
                is_preempting = True  
                fields = data.split(" ")
                if len(fields) > 1:
                    notbefore = fields[-1]
                    notbefore = datetime.strptime(notbefore,"%a,_%d_%b_%Y_%H:%M:%S_%Z")
                    sleep_time = (notbefore - datetime.now()).seconds - 30
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                print('Trigger preempt!', flush=True)
                send_signal()
            else:
                print('preempt already triggered!',flush=True)
            Handler.triggermorph.release()
        
        elif 'checkpoint done' in data:
            Handler.trackcheckpoints.acquire()
            Handler.triggermorph.acquire()
            last_iter = int(str(data).split(" ")[-1])
            if is_preempting:
                print('Preempt successful {}'.format(last_iter), flush=True)
                Handler.notify()
                time.sleep(120)     # wait for scheduled event to occur 
                Handler.kill_all()
                curr_world_size = 0
                Handler.update_available()
                Handler.start_remote()
                is_preempting = False
            elif is_morphing:
                print("Morph successful {}".format(last_iter), flush=True)
                Handler.kill_all()
                curr_world_size = 0
                Handler.start_remote(last_iter)
                is_morphing = False
                is_restarting = False
            elif not is_restarting:
                if last_ckpt_signal is None or \
                (recv_time - last_ckpt_signal).total_seconds() > 120:
                    print("Handling restart", last_ckpt_signal)
                    last_iter = int(str(data).split(" ")[-1])
                    Handler.notify()
                    time.sleep(120)    # wait for transient errors to pass
                    Handler.kill_all()
                    curr_world_size = 0
                    Handler.update_available()
                    Handler.start_remote(last_iter)
                    is_restarting = False
            last_ckpt_signal = recv_time
            Handler.triggermorph.release()
            Handler.trackcheckpoints.release()
        
        elif 'morph' in data:
            Handler.triggermorph.acquire()
            Handler.trackcheckpoints.acquire()
            if not is_preempting and not is_restarting and not is_morphing:
                print("Morphing!",flush=True)
                is_restarting = True
                is_morphing = True
                response = Handler.send_signal()
                Handler.update_available()
                if curr_world_size == 0:
                    print("Nothing running currently, will start")
                    Handler.start_remote(last_iter)
            else:
                print("morph change was already detected", is_morphing, is_preempting, is_restarting)
            Handler.trackcheckpoints.release()
            Handler.triggermorph.release()
        print("handle done", flush=True)
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "10.0.3.4", 4200

    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    with server:
        server.serve_forever()