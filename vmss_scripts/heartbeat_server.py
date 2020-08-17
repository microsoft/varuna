# to be run in manager
import socket
import threading
import socketserver
import time
from datetime import datetime
import os
import sys

num_workers = int(sys.argv[1])

last_heartbeat_time = datetime.now()
step_beat_map = dict()
completed_steps = 0

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    triggermorph = threading.Lock()
    trackcheckpoints = threading.Lock()

    def handle(self):
        global last_heartbeat_time, step_count_map
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        print("{} got something from {}: {}".format(datetime.now(), self.client_address, data), flush=True)
        
        if "start" in data:
            world_size = int(data.split(" ")[-1])
            for i in range(world_size):
                step_count_map.append(0)
        if "step" in data:
            rank, step = data.split(" ")[-2:]
            rank = int(rank); step = int(step)
            if completed_steps < step:
                step_beat_map[rank] = True
            if len(step_beat_map)
        print("handle done", flush=True)
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "172.16.5.4", 5000

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    
    with server:
        server.serve_forever()