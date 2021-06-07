import socket
from datetime import datetime
import os
import time
import sys
from random import randint

# cluster = sys.argv[1]
counter = 0
possibly_dead_nodes = []

available_machines_list = sys.argv[1]
running_machines_list = sys.argv[2]

server_ip = sys.argv[3]
server_port = sys.argv[4]

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))

def poll_and_update():
    print(str(datetime.now()), flush=True)
    current_machines = get_current_machines()
    current_num_machines = len(current_machines)
    print("Current:", current_machines,flush=True)

    new_machines = get_available_machines()
    print("New", new_machines, flush=True)

    if sorted(new_machines) == sorted(current_machines):
        print("no morph", flush=True)
    else:
        # machines_added = [m for m in new_machines if m not in current_machines]
        msg = f"morph {len(new_machines)}"
        client(server_ip, server_port, msg)
        print(len(new_machines), flush=True)


def get_current_machines():
    f = open(running_machines_list,"r")
    machines = f.read().split("\n")
    machines = [m for m in machines if len(m) > 0]
    return machines

def get_available_machines():
    f = open(available_machines_list,"r")
    machines = f.read().split("\n")
    machines = [m for m in machines if len(m) > 0]
    return machines

if __name__ == "__main__":

    # while True:
    poll_and_update()
    # time.sleep(5*60)

