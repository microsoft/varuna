# to be run in manager
import socket
import threading
import socketserver
import time

from datetime import datetime
import os
import subprocess
import sys

morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

num_nodes = 0
compute_times = dict()
all_times = []
all_ips = []


class Handler(socketserver.BaseRequestHandler):
    static_lock = threading.Lock()

    def handle(self):
        global all_times, all_ips
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)
        Handler.static_lock.acquire()
        all_times.append(float(data))
        all_ips.append(self.client_address)
        Handler.static_lock.release()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "10.0.3.4", 4100


    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()

    # wait for all workers
    time.sleep(180)

    server.shutdown()
    server.server_close()

    print("total workers:", len(all_times), len(all_ips))
    avg_time = sum(all_times)/len(all_times)
    min_time = min(all_times)
    max_time = max(all_times)
    print("min: ", min_time, ", max: ", max_time, ", avg: ", avg_time )
    bad_ips = set()
    if max_time - min_time > 10:
        for i,t in enumerate(all_times):
            if t > min_time + 10:
                bad_ips.add(all_ips[i])

    print("Slow ips are:", bad_ips)

    with open(os.path.join(morph_path, "slow_machines.out"), "w") as f:
        for ip in bad_ips:
            f.write(ip + "\n")
    # with server:
    #     server.serve_forever()