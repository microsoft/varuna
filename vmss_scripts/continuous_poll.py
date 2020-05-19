import socket
from datetime import datetime
import os
import time
from random import randint

counter = 0
possibly_dead_nodes = []

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))

def poll_and_update():
    print(str(datetime.now()), flush=True)
    current_machines = get_current_machines()
    current_num_machines = len(current_machines)
    print("Current:", current_machines)

    new_machines = get_available_machines()
    print("New", new_machines)

    if sorted(new_machines) == sorted(current_machines):
        print("no morph", flush=True)
    else:
        client(server_ip, server_port, "morph")
        print(len(new_machines), flush=True)

def get_current_machines():
    f = open("/home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out","r")
    machines = f.read().split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    return machines

def get_available_machines():
    # gets reachable machines
    bash_out = os.popen("bash /home/varuna/t-nisar/Varuna/Megatron-LM/get_available_machines.sh").read()
    machines = bash_out.split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    return machines

if __name__ == "__main__":

    server_ip = "172.16.5.4"
    server_port = 4200

    while True:
        poll_and_update()
        time.sleep(900)

