# to be run in manager
import socket
import threading
import socketserver
import time
from datetime import datetime
import os
import sys
from threading import Thread
from collections import defaultdict 

last_heartbeat_time = datetime.now()
completed_steps = 0

ckpt_dir = "/home/varuna/gpt2-blob/morph_stress_test"
morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

class Handler(socketserver.BaseRequestHandler):

    step_lock = threading.Lock()

    def handle(self):
        global last_heartbeat_time, completed_steps
        data = str(self.request.recv(1024), 'ascii')
        print("{} got something from {}: {}".format(datetime.now(), self.client_address, data), flush=True)
        cur_thread = threading.current_thread()

        if "progress" in data:
            Handler.step_lock.acquire()
            try:
                step = int(data.split(" ")[-1])
                completed_steps = step
                last_heartbeat_time = datetime.now()
            except Exception as e:
                print("Caught exception while stepping", e)
            Handler.step_lock.release()
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def check_progress():
    global completed_steps
    last_checked_iter = 0
    while True:
        try:
            if last_checked_iter == completed_steps:
                print('{}: Training stuck at {}. Restarting!'.format(datetime.now(), last_checked_iter), flush=True)    
                
                os.system("sudo kill -9 $(ps aux |grep morph | awk -F ' ' '{print $2}')")
                os.system("sudo kill -9 $(ps aux |grep continuous_poll | awk -F ' ' '{print $2}')")
                os.chdir(morph_path)
                os.system("bash kill_all.sh")

                all_ckpt = [int(f.split("_")[-1]) for f in os.listdir(ckpt_dir) if "opt_ckpt" in f]
                all_ckpt = sorted(all_ckpt)
                if len(all_ckpt) > 0:
                    last_ckpt = all_ckpt[-1]
                else:
                    last_ckpt = -1
                print("last ckpt is", last_ckpt)
                
                os.chdir("..")
                os.system("python3 vmss_scripts/morph_server.py 0 {} > morph.out 2>morph.err &".format(last_ckpt))
                # reboot + remount etc.
                open(os.path.join(morph_path,"available_machines.out"), "w")
                os.system("python3 vmss_scripts/continuous_poll.py > poll.out 2>poll.err &")
                print("Restart done!", flush=True)
            else:
                print(datetime.now(),"Got timely update!", completed_steps)
            last_checked_iter = completed_steps
        except Exception as e:
            print("Caught exception in progress thread:", e)
        time.sleep(60*30)

if __name__ == "__main__":
    HOST, PORT = "10.0.3.4", 5000

    server = ThreadedTCPServer((HOST, PORT), Handler)

    check_progress_thread = Thread(target=check_progress, args=())
    check_progress_thread.daemon=True
    check_progress_thread.start()
    
    with server:
        server.serve_forever()