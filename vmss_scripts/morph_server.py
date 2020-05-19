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
num_running_nodes = 0

class Handler(socketserver.BaseRequestHandler):

    triggermorph = threading.Lock()
    trackcheckpoints = threading.Lock()

    def handle(self):
        global checkpointed, is_preempting, num_running_nodes, is_restarting, is_morphing, last_ckpt_signal
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)
        if 'preempt' in data:
            Handler.triggermorph.acquire()
            if not is_morphing and not is_preempting:
                # set False to ignore signals from other VMs, set True after checkpointing succeeds
                is_preempting = True  
                fields = data.split(" ")
                if len(fields) > 1:
                    notbefore = fields[-1]
                    notbefore = datetime.strptime(notbefore,"%a,_%d_%b_%Y_%H:%M:%S_%Z")
                    sleep_time = (notbefore - datetime.now()).seconds - 30
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                #get number of machines
                # num_running_nodes = int(open("/home/varuna/t-nisar/Varuna/Megatron-LM/nservers").read())       
                print('Trigger morph!', flush=True)
                os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/send_signal.sh")                
            else:
                print('preempt already triggered!',flush=True)
            Handler.triggermorph.release()
        elif 'checkpoint done' in data:
            Handler.trackcheckpoints.acquire()
            checkpointed += 1
            if checkpointed == 1:
                if is_preempting:
                    last_iter = int(str(data).split(" ")[-1])
                    print('Checkpoint successful {}'.format(last_iter), flush=True)
                    handle_request = True
                    # wait for scheduled event to occur 
                    os.system('(echo "Subject: Job morphing"; echo "the job has stopped :/") | ssmtp -F "Varuna" "nitika.saran@gmail.com, t-saathl@microsoft.com"')
                    time.sleep(150)
                    # double checking that all pretraining processes are killed 
                    # get available machines
                    os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/get_available_machines.sh > /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out")
                    # - not clean and should be removed ideally
                    os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/kill_all.sh")
                    # resume model in available machines
                    os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/start_remote.sh {}".format(last_iter))
                    is_preempting = False
                elif is_morphing:
                    print("morph procedure")
                    last_iter = int(str(data).split(" ")[-1])
                    os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/kill_all.sh")
                    os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/start_remote.sh {}".format(last_iter))
                    is_morphing = False
                elif not is_restarting:
                    if last_ckpt_signal is None or \
                    (recv_time - last_ckpt_signal).seconds > 120:
                        print("Handling restart", last_ckpt_signal)
                        last_ckpt_signal = recv_time
                        last_iter = int(str(data).split(" ")[-1])
                        is_restarting = True
                        os.system('(echo "Subject: Job stopped"; echo "the job has stopped :/") | ssmtp -F "Varuna" "nitika.saran@gmail.com, t-saathl@microsoft.com"')
                        time.sleep(120)
                        os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/get_available_machines.sh > /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out")
                        os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/kill_all.sh")
                        os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/start_remote.sh {}".format(last_iter))
                        is_restarting = False
                        # restarted = True
                checkpointed = 0
            Handler.trackcheckpoints.release()
        elif 'morph' in data:
            Handler.triggermorph.acquire()
            Handler.trackcheckpoints.acquire()
            if not is_preempting and not is_restarting and not is_morphing:
                print("Morphing!")
                is_restarting = True
                is_morphing = True
                os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/get_available_machines.sh > /home/varuna/t-nisar/Varuna/Megatron-LM/available_machines.out")
                response = os.system("bash /home/varuna/t-nisar/Varuna/Megatron-LM/send_signal.sh")
                print("send signal response", response, flush=True)
            else:
                print("morph change was already detected")
            Handler.trackcheckpoints.release()
            Handler.triggermorph.release()
        print("handle done", flush=True)
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "172.16.5.4", 4200

    server = ThreadedTCPServer((HOST, PORT), Handler)
    
    with server:
        server.serve_forever()