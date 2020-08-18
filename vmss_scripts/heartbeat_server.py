# to be run in manager
import socket
import threading
import socketserver
import time
from datetime import datetime
import os
import sys
from collections import defaultdict 

num_workers = int(sys.argv[1])

last_heartbeat_time = datetime.now()
step_beat_map = defaultdict(lambda: [])
seen_map = dict()
completed_steps = 0
worker_count = 0
ongoing_step = -1

class Handler(socketserver.BaseRequestHandler):

    step_lock = threading.Lock()

    def handle(self):
        global last_heartbeat_time, num_workers, completed_steps, worker_count, step_beat_map, ongoing_step
        data = str(self.request.recv(1024), 'ascii')
        print("{} got something from {}: {}".format(datetime.now(), self.client_address, data), flush=True)
        cur_thread = threading.current_thread()
        return 
        if "start" in data:
            world_size = int(data.split(" ")[-1])
            # for i in range(world_size):
            #     step_count_map.append(0)
            num_workers = world_size

        if "step" in data:
            Handler.step_lock.acquire()
            try:
                _, step, rank, stage, fwd_time = data.split(" ")
                stage = int(stage); step = int(step); 
                fwd_time = float(fwd_time)
                already_seen = rank in seen_map and seen_map[rank] == True
                if not already_seen and completed_steps < step and worker_count < num_workers:
                    if ongoing_step == -1:
                        ongoing_step = step
                    step_beat_map[stage].append(fwd_time)
                    seen_map[rank] = True
                    worker_count += 1
                    print("worker count", worker_count)
                    if worker_count == num_workers:
                        print("Finished step", step)
                        for s in step_beat_map:
                            a = sum(step_beat_map[s]) / len(step_beat_map)
                            mi = min(step_beat_map[s])
                            ma = max(step_beat_map[s])
                            print("min,max,avg time for stage", s, ": ", mi, ma, a, flush=True)
                        step_beat_map = defaultdict(lambda: [])
                        for r in seen_map:
                            seen_map[r] = False
                        worker_count = 0
                        completed_steps = ongoing_step
                        ongoing_step = -1
            except Exception as e:
                print("Caught exception while stepping", e)
            Handler.step_lock.release()
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "10.0.3.4", 5000

    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    with server:
        server.serve_forever()