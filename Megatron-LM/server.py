import socket
import threading
import socketserver

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        print("got something from ", self.client_address)
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print("Received: {}".format(response))

if __name__ == "__main__":

    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 50007

    client(HOST, PORT, "Hello World 1")
    client(HOST, PORT, "Hello World 2")
    client(HOST, PORT, "Hello World 3")
    exit()

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        server.serve_forever()
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        # server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        # server_thread.daemon = True
        # server_thread.start()
        print("Server loop running in thread:", server_thread.name)
        # server_thread.join()

        client(ip, port, "Hello World 1")
        client(ip, port, "Hello World 2")
        client(ip, port, "Hello World 3")

        # server.shutdown()