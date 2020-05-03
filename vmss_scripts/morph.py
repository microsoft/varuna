# to be run in manager
import socket
import threading
import socketserver
import time

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    handle_request = True
    checkpointed = 0
    triggermorph = threading.Lock()
    trackcheckpoints = threading.Lock()
    num_nodes = 32 #get number of machines

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        if 'morph' in data:
            self.triggermorph.acquire()
            if self.handle_request:
                self.handle_request = False         # set False to ignore signals from other VMs, set True after checkpointing succeeds
                print('Trigger checkpointing!')
                exec("/home/varuna/vmss-scripts/send_signal.sh "+str(self.num_nodes))       # trigger checkpointing in all nodes
            else:
                print('Checkpoint already triggered!')
            self.triggermorph.release()
        if 'checkpoint done' in data:
            self.trackcheckpoints.acquire()
            self.checkpointed += 1
            if self.checkpointed == self.num_nodes:
                print('Checkpoint successful in all nodes')
                self.handle_request = True

                # resume model in available machines
                # get number of machines available
                #exec("home/varuna/vmss-scripts/start_remote.sh"+)
            self.trackcheckpoints.release()
        if 'checkpoint failed' in data:
            print('checkpoint failed in ', self.client_address[0])
            

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    HOST, PORT = "172.16.5.4", 4200

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # time.sleep(200)
        # server.shutdown()