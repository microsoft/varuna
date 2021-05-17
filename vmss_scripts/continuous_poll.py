import socket
from datetime import datetime
import os
import time
import sys
from random import randint

# cluster = sys.argv[1]
counter = 0
possibly_dead_nodes = []

morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

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
        # print(" ".join(new_machines))
        print(len(new_machines), flush=True)

def run_cmd_all(cmd, machines):
    for m in machines:
        os.system("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                    varuna@{} -i /home/varuna/.ssh/vdummy.pem \
                    \"{}\"".format(m, cmd))

def get_current_machines():
    machines_list = os.path.join(morph_path, "available_machines.out")
    f = open(machines_list,"r")
    machines = f.read().split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    return machines

def get_available_machines():
    # gets reachable machines
    ping_script = os.path.join(morph_path,"get_available_machines.sh")
    bash_out = os.popen("bash {}".format(ping_script)).read()
    machines = bash_out.split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    return machines

if __name__ == "__main__":

    server_ip = "10.0.0.4"
    server_port = 4200

    while True:
        poll_and_update()
        time.sleep(5*60)

