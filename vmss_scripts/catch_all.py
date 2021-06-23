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
last_iter = 18750
sleep_time_interval=30*60
# was_stuck = True

# will check for 10 steps
slowcheck_dict = [[] for _ in range(10)]

ckpt_dir = "/home/varuna/gpt2-blob/mega_2_5b_4.8e-3_clipgrads"
morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

class Handler(socketserver.BaseRequestHandler):

    step_lock = threading.Lock()

    def handle(self):
        global last_heartbeat_time, completed_steps, last_iter, sleep_time_interval #, was_stuck
        data = str(self.request.recv(1024), 'ascii')
        # print("{} got something from {}: {}".format(datetime.now(), self.client_address, data), flush=True)
        cur_thread = threading.current_thread()

        if "progress" in data:
            Handler.step_lock.acquire()
            try:
                print(f"{datetime.now()} Got step update from {self.client_address}: {data}", flush=True)
                _, batch_time, step = data.split(" ")
                step = int(step); batch_time = float(batch_time)
                # if was_stuck:
                sleep_time_interval = int( (batch_time * 5 * 2) + (60*10) )
                print("batch time is",batch_time)
                print("setting wait interval to ", sleep_time_interval/60,"minutes", flush=True)
                # was_stuck = False
                completed_steps = step
                last_heartbeat_time = datetime.now()
            except Exception as e:
                print("Caught exception while stepping", e, flush=True)
            Handler.step_lock.release()

            if completed_steps >= last_iter:
                print("Training complete! Exiting")
                os.system("sudo kill -9 $(ps aux |grep morph | awk -F ' ' '{print $2}')")
                os.system("sudo kill -9 $(ps aux |grep continuous_poll | awk -F ' ' '{print $2}')")                
                exit()
        
        if "slowcheck" in data:
            Handler.step_lock.acquire()
            try:
                global slowcheck_dict
                _, step_num, stage, dp_rank, fwd_time = data.split(" ")
                stage = int(stage); fwd_time = float(fwd_time)
                step_num = int(step_num)
                ip = self.client_address[0]
                if step_num < len(slowcheck_dict):
                    first = len(slowcheck_dict[step_num]) == 0
                    slowcheck_dict[step_num].append((stage, fwd_time, ip))
            except Exception as e:
                print("Caught exception while checking", e, flush=True)
            Handler.step_lock.release()
            if first:
                time.sleep(30)
                stage_fwd_time = dict()
                for stage, fwd_time, ip in slowcheck_dict[step_num]:
                    if stage not in stage_fwd_time:
                        stage_fwd_time[stage] = []
                    stage_fwd_time[stage].append((fwd_time, ip))
                slow_ips = set()
                print(f"Step {step_num}")
                for stage in stage_fwd_time:
                    print(f"Stage {stage}: {stage_fwd_time[stage]}")
                    times = sorted([t for t,ip in stage_fwd_time[stage]])
                    dp = len(times)
                    mid_i = dp // 2
                    median_time = times[mid_i]
                    if dp % 2 == 0:
                        median_time = (median_time + times[mid_i-1]) / 2
                    for t, ip in stage_fwd_time[stage]:
                        excess = (t-median_time)/median_time
                        if excess > 0.15:
                            slow_ips.add(ip)
                print("Slow IPs:", slow_ips, flush=True)
                if step_num > 1:
                    slow_filename = os.path.join(morph_path,"slow_machines.out")
                    with open(slow_filename,"r") as f:
                        slow_prev = f.read().split("\n")
                        for ip in slow_prev:
                            slow_ips.add(ip)
                    slow_out = open(slow_filename, "w")
                    for ip in slow_ips:
                        slow_out.write(f"{ip}\n")
                    slow_out.close()
                Handler.step_lock.acquire()
                slowcheck_dict[step_num] = []
                Handler.step_lock.release()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def check_progress():
    global completed_steps, sleep_time_interval
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
                
                os.chdir(morph_path)
                os.chdir("..")
                os.system("python3 vmss_scripts/morph_server.py 0 {} > morph.out 2>morph.err &".format(last_ckpt))
                # reboot + remount etc.
                open(os.path.join(morph_path,"available_machines.out"), "w")
                os.system("python3 vmss_scripts/continuous_poll.py > poll.out 2>poll.err &")
                print("resetting wait interval")
                sleep_time_interval = 30*60
                # was_stuck = True
                print("Restart done!", flush=True)
            else:
                print(datetime.now(),"Got timely update!", completed_steps, flush=True)
            last_checked_iter = completed_steps
        except Exception as e:
            print("Caught exception in progress thread:", e, flush=True)
        time.sleep(sleep_time_interval)

if __name__ == "__main__":
    HOST, PORT = "10.0.0.4", 5000

    server = ThreadedTCPServer((HOST, PORT), Handler)

    check_progress_thread = Thread(target=check_progress, args=())
    check_progress_thread.daemon=True
    check_progress_thread.start()
    
    with server:
        server.serve_forever()